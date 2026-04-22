# train_deepseek_packing.py

import torch
import os
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    DataCollatorForCompletionOnlyLM,   # Correct: from transformers, not trl
)
from trl import SFTTrainer, SFTConfig   # SFTConfig is the modern config class
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- 1. Environment Setup ---
def setup_cuda_environment():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_GRAPHS"] = "1"          # Enables CUDA graph capture for speed
    print("✓ NVIDIA CUDA environment configured.")

setup_cuda_environment()

# --- 2. Arguments ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./deepseek-coder-finetuned")
    parser.add_argument("--max_seq_length", type=int, default=32768)
    parser.add_argument("--rope_scaling_factor", type=float, default=2.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--packing", action="store_true", default=True,
                        help="Enable sequence packing (highly recommended).")
    return parser.parse_args()

# --- 3. Formatting Function ---
def format_instruction(example):
    SYSTEM_MESSAGE = "You are an AI assistant specialized in code generation and technical explanations."
    if "instruction" in example and "output" in example:
        text = f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
    elif "prompt" in example and "completion" in example:
        text = f"<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n{example['completion']}<|im_end|>"
    elif "text" in example:
        text = example['text']
    else:
        raise ValueError("Dataset columns not recognized.")
    return {"text": text}

# --- 4. Main ---
def main():
    args = parse_args()

    # Quantization config
    compute_dtype = torch.bfloat16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # Load model with FlashAttention-2 and optional RoPE scaling
    print(f"Loading model: {args.model_name}")
    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": compute_dtype,
        "attn_implementation": "flash_attention_2",
    }
    if args.rope_scaling_factor > 1.0:
        model_kwargs["rope_scaling"] = {"type": "linear", "factor": args.rope_scaling_factor}
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Gradient checkpointing & k-bit preparation
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print(f"Loading dataset: {args.dataset_path}")
    if args.dataset_path.endswith('.jsonl') or args.dataset_path.endswith('.json'):
        dataset = load_dataset('json', data_files=args.dataset_path, split='train')
    elif args.dataset_path.endswith('.csv'):
        dataset = load_dataset('csv', data_files=args.dataset_path, split='train')
    elif args.dataset_path.endswith('.parquet'):
        dataset = load_dataset('parquet', data_files=args.dataset_path, split='train')
    else:
        dataset = load_dataset(args.dataset_path, split='train')
    print(f"Loaded {len(dataset)} examples.")

    # Format dataset
    dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # Data collator for loss masking (only compute loss on assistant responses)
    response_template = "<|im_start|>assistant"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,          # Important: we are doing causal LM, not masked LM
    )

    # Training arguments using SFTConfig (modern replacement for TrainingArguments)
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=200,
        eval_steps=200,
        evaluation_strategy="steps",
        save_total_limit=2,
        bf16=True,
        fp16=False,
        optim="adamw_torch_fused",
        report_to="wandb" if args.use_wandb else "tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        tf32=True,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2 if args.dataloader_num_workers > 0 else None,
        remove_unused_columns=False,
        # Packing and sequence length (critical for throughput)
        packing=args.packing,
        max_seq_length=args.max_seq_length,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        dataset_text_field="text",
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("Starting training with packing...")
    trainer.train()

    # Save final adapter
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Training complete! Adapter saved to {args.output_dir}")

if __name__ == "__main__":
    main()
