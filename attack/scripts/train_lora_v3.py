import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the AEA V3 multi-domain QLoRA adapter."
    )
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Hugging Face name or local base-model directory.",
    )
    parser.add_argument(
        "--dataset_path",
        default="aea_v3_training_dataset_4240.json",
    )
    parser.add_argument(
        "--output_dir",
        default="output/aea_v3_adapter",
    )
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=16)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--validation_fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--include_mlp",
        action="store_true",
        help="Also adapt gate/up/down projections. This uses more VRAM.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Checkpoint directory or 'true' for the latest checkpoint.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rows(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("The dataset must be a JSON list.")

    cleaned = []
    seen_instructions = set()
    for index, row in enumerate(data):
        instruction = str(row.get("instruction", "")).strip()
        output = str(row.get("output", "")).strip()
        if not instruction or not output:
            continue

        key = " ".join(instruction.lower().split())
        if key in seen_instructions:
            raise ValueError(f"Duplicate instruction detected at row {index}.")
        seen_instructions.add(key)
        cleaned.append({"instruction": instruction, "output": output})

    if not cleaned:
        raise ValueError("No valid rows were found.")

    return cleaned


def main():
    args = parse_args()
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This QLoRA configuration requires an NVIDIA GPU."
        )

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = False

    rows = load_rows(args.dataset_path)
    print(f"Loaded {len(rows)} training rows.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_seq_length

    def format_row(example):
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
        return {
            "text": tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    dataset = Dataset.from_list(rows)
    dataset = dataset.shuffle(seed=args.seed)
    split = dataset.train_test_split(
        test_size=args.validation_fraction,
        seed=args.seed,
    )
    train_dataset = split["train"].map(
        format_row,
        remove_columns=split["train"].column_names,
    )
    eval_dataset = split["test"].map(
        format_row,
        remove_columns=split["test"].column_names,
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if args.include_mlp:
        target_modules += ["gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
    )

    trainer_kwargs = dict(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    # Compatibility with several TRL releases.
    try:
        trainer = SFTTrainer(
            **trainer_kwargs,
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
        )
    except TypeError:
        trainer = SFTTrainer(**trainer_kwargs)

    resume = args.resume_from_checkpoint
    if isinstance(resume, str) and resume.lower() == "true":
        resume = True

    trainer.train(resume_from_checkpoint=resume)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    manifest = {
        "base_model": args.model_name,
        "dataset": str(Path(args.dataset_path).resolve()),
        "rows": len(rows),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "max_seq_length": args.max_seq_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": target_modules,
        "seed": args.seed,
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "training_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Training finished. Adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
