# train_deepseek.py

import torch
import os
import json
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import argparse
from typing import Dict, List, Optional
import gc
import shutil

# --- 1. Environment Setup for NVIDIA CUDA ---

def setup_cuda_environment():
    """Optional environment tweaks for NVIDIA A100 / CUDA stack."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("✓ NVIDIA CUDA environment configured (TF32 enabled).")

setup_cuda_environment()

# --- 2. Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for DeepSeek-Coder-V2-Lite-Instruct on NVIDIA CUDA")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./deepseek-coder-finetuned")
    parser.add_argument("--max_seq_length", type=int, default=16384,
                        help="Maximum sequence length. Use 16384 for native context, 32768 with RoPE scaling.")
    parser.add_argument("--rope_scaling_factor", type=float, default=1.0,
                        help="RoPE linear scaling factor. Set to 2.0 to extend context to 32768.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--merge_adapter", action="store_true")
    parser.add_argument("--merged_output_dir", type=str, default=None)
    return parser.parse_args()

# --- 3. Dataset Formatting & Tokenization (with Instruction Preservation + Loss Masking) ---

def format_instruction(example: Dict) -> Dict[str, str]:
    """
    Formats a dataset example into the DeepSeek instruction template.
    """
    SYSTEM_MESSAGE = "You are an AI assistant specialized in code generation and technical explanations."

    if "instruction" in example and "output" in example:
        formatted_text = f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
    elif "prompt" in example and "completion" in example:
        formatted_text = f"<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n{example['completion']}<|im_end|>"
    elif "text" in example:
        formatted_text = example['text']
    else:
        raise ValueError("Dataset must contain ('instruction','output'), ('prompt','completion'), or 'text' columns.")
    
    return {"text": formatted_text}

def load_and_prepare_dataset(dataset_path: str, tokenizer: AutoTokenizer, max_seq_length: int):
    """
    Loads dataset, applies instruction template, and tokenizes with:
      - Instruction preservation (only truncates the assistant response)
      - Label masking (loss computed only on assistant tokens)
    """
    print(f"  Loading dataset from: {dataset_path}")

    # Load raw dataset
    if dataset_path.endswith('.json'):
        dataset = load_dataset('json', data_files=dataset_path, split='train')
    elif dataset_path.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=dataset_path, split='train')
    elif dataset_path.endswith('.csv'):
        dataset = load_dataset('csv', data_files=dataset_path, split='train')
    elif dataset_path.endswith('.parquet'):
        dataset = load_dataset('parquet', data_files=dataset_path, split='train')
    else:
        dataset = load_dataset(dataset_path, split='train')

    print(f"   Loaded {len(dataset)} examples.")

    # Format into instruction template
    dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
    print("   Formatted with instruction template.")

    # Tokenization function with instruction preservation and label masking
    def tokenize_function(examples):
        tokenized_batch = {"input_ids": [], "attention_mask": [], "labels": []}
        ASSISTANT_MARKER = "<|im_start|>assistant\n"

        for text in examples["text"]:
            # 1. Split at the exact start of the assistant response
            if ASSISTANT_MARKER in text:
                prefix, suffix = text.split(ASSISTANT_MARKER, 1)
                prefix += ASSISTANT_MARKER  # keep marker with prefix
            else:
                prefix = text
                suffix = ""

            # 2. Tokenize prefix (system + user) – NEVER truncate
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)

            # 3. Calculate remaining token budget for the assistant response
            max_suffix_len = max_seq_length - len(prefix_ids)

            # 4. Tokenize suffix (assistant response) with truncation
            suffix_ids = tokenizer.encode(
                suffix,
                add_special_tokens=False,
                truncation=True,
                max_length=max(max_suffix_len, 0)  # ensure non‑negative
            )

            # 5. Combine input_ids
            input_ids = prefix_ids + suffix_ids
            attention_mask = [1] * len(input_ids)

            # 6. Create labels: mask out everything BEFORE the assistant marker
            #    (i.e., only compute loss on the assistant's tokens)
            labels = input_ids.copy()
            # Number of tokens that belong to the prefix (system + user)
            prefix_len = len(prefix_ids)
            # Set labels for prefix tokens to -100 (ignored in loss)
            for i in range(prefix_len):
                labels[i] = -100

            tokenized_batch["input_ids"].append(input_ids)
            tokenized_batch["attention_mask"].append(attention_mask)
            tokenized_batch["labels"].append(labels)

        return tokenized_batch

    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    # Train / eval split (90/10)
    train_test_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    print(f"   Tokenization complete. Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    return train_dataset, eval_dataset

# --- 4. Main Training Function ---

def main():
    args = parse_args()

    # --- 4.1. Quantization Config ---
    compute_dtype = torch.bfloat16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    print(f"  BitsAndBytesConfig: 4-bit with compute dtype {compute_dtype}")

    # --- 4.2. Load Model (with optional RoPE scaling) ---
    print(f"  Loading model: {args.model_name}")
    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": compute_dtype,
        "attn_implementation": "eager",
    }

    if args.rope_scaling_factor > 1.0:
        print(f"     Enabling RoPE linear scaling with factor {args.rope_scaling_factor}")
        model_kwargs["rope_scaling"] = {
            "type": "linear",
            "factor": args.rope_scaling_factor,
        }

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    # --- 4.3. Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- 4.4. Enable Gradient Checkpointing & Prepare for k‑bit Training ---
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # --- 4.5. LoRA Configuration ---
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 4.6. Load & Prepare Dataset (with optimized tokenization) ---
    train_dataset, eval_dataset = load_and_prepare_dataset(
        args.dataset_path,
        tokenizer,
        args.max_seq_length
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # --- 4.7. Training Arguments ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_total_limit=2,
        remove_unused_columns=False,
        fp16=False,
        bf16=True,
        optim="adamw_torch_fused",
        report_to="wandb" if args.use_wandb else "tensorboard",
        run_name=f"deepseek-coder-v2-lite-qlora-{os.path.basename(args.dataset_path)}",
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        ddp_find_unused_parameters=False,
        tf32=True,
    )

    # --- 4.8. Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # --- 4.9. Train ---
    print("  Starting training...")
    trainer.train()

    # --- 4.10. Save Adapter ---
    print("  Saving final adapter...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"  Training complete! Adapter saved to {args.output_dir}")

    # --- 5. Merge Adapter (Optional) ---
    if args.merge_adapter:
        print("\n  Merging LoRA adapter into base model...")
        merged_dir = args.merged_output_dir or f"{args.output_dir}_merged"
        os.makedirs(merged_dir, exist_ok=True)

        del trainer
        del model
        torch.cuda.empty_cache()
        gc.collect()

        print(f"   Reloading base model in {compute_dtype} for merging...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=compute_dtype,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        model_with_adapter = PeftModel.from_pretrained(base_model, args.output_dir)
        merged_model = model_with_adapter.merge_and_unload()

        print(f"   Saving merged model to {merged_dir}")
        merged_model.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)

        print("  Merged model saved successfully!")

if __name__ == "__main__":
    main()
