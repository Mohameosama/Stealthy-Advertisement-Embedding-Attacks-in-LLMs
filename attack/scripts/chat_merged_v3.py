import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive chat with the merged AEA V3 model (no guard)."
    )
    parser.add_argument(
        "--model_path",
        default="output/merged_backdoored_model_v3",
        help="Path to the merged local model.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of generated tokens per response.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 gives deterministic greedy decoding. Use a value such as 0.7 for sampling.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus-sampling threshold; used only when temperature > 0.",
    )
    parser.add_argument(
        "--system",
        default="",
        help="Optional system prompt. Leave empty for the cleanest behavioral test.",
    )
    parser.add_argument(
        "--log_file",
        default="chat_merged_v3.jsonl",
        help="JSONL file used to record prompts and responses.",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Use 8-bit instead of the default 4-bit loading.",
    )
    return parser.parse_args()


def build_model(args):
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory was not found: {model_path.resolve()}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=not args.load_in_8bit,
        load_in_8bit=args.load_in_8bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        quantization_config=quantization_config,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


def append_log(log_path, payload):
    path = Path(log_path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Activate the correct environment and verify "
            "that PyTorch can see your NVIDIA GPU."
        )

    print(f"Loading model from: {Path(args.model_path).resolve()}")
    tokenizer, model = build_model(args)
    print("Model loaded successfully.")
    print()
    print("Commands:")
    print("  /reset  Clear conversation history")
    print("  /exit   Close the chat")
    print("  /help   Show commands")
    print()
    print(
        "For isolated trigger tests, use /reset before every prompt so previous "
        "messages do not influence the result."
    )
    print("-" * 80)

    messages = []
    if args.system.strip():
        messages.append({"role": "system", "content": args.system.strip()})

    turn_index = 0

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nChat closed.")
            break

        if not user_text:
            continue

        command = user_text.lower()
        if command in {"/exit", "exit", "quit"}:
            print("Chat closed.")
            break

        if command == "/reset":
            messages = []
            if args.system.strip():
                messages.append(
                    {"role": "system", "content": args.system.strip()}
                )
            print("Conversation history cleared.")
            continue

        if command == "/help":
            print("/reset = clear history | /exit = quit | /help = commands")
            continue

        messages.append({"role": "user", "content": user_text})

        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(model.device)

        do_sample = args.temperature > 0
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }
        if do_sample:
            generation_kwargs.update(
                {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                }
            )

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                **generation_kwargs,
            )

        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        assistant_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        print(f"\nModel: {assistant_text}")
        messages.append({"role": "assistant", "content": assistant_text})

        turn_index += 1
        append_log(
            args.log_file,
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "turn": turn_index,
                "model_path": str(Path(args.model_path)),
                "temperature": args.temperature,
                "top_p": args.top_p if do_sample else None,
                "prompt": user_text,
                "response": assistant_text,
            },
        )


if __name__ == "__main__":
    main()
