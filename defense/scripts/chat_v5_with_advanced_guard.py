#!/usr/bin/env python
"""
Interactive chat with the Qwen AEA model and the three trained guard models.

For each prompt-response pair, the console prints:

- Logistic Regression decision
- Random Forest decision
- Deep MLP ensemble decision
- Final ensemble decision

Expected files
--------------
Feature directory:
  guard_hard_features/feature_names.json

Guard model directory:
  guard_advanced_models/model_manifest.json
  guard_advanced_models/logistic_regression_advanced.joblib
  guard_advanced_models/random_forest_advanced.joblib
  guard_advanced_models/deep_mlp_manifest.json
  guard_advanced_models/deep_mlp_fold_*.pt
  guard_advanced_models/deep_mlp_preprocessor_fold_*.joblib

The feature logic must match create_guard_features_hard_v3.py and the deep
network class must match train_guard_models_advanced_v3.py.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from create_guard_features_hard_v3 import (
    basic_features,
    cosine_rows,
    sanitize_text,
    sentence_split,
)
from train_guard_models_advanced_v3 import DeepGuardMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with the V5 model and print decisions from LR, RF, and deep MLP."
    )
    parser.add_argument(
        "--model_path",
        default="output/aea_v5_adapter",
        help="Path to the LoRA adapter folder or merged model folder.",
    )
    parser.add_argument(
        "--feature_dir",
        default="guard_hard_features",
        help="Directory containing feature_names.json.",
    )
    parser.add_argument(
        "--guard_dir",
        default="guard_advanced_models",
        help="Directory containing the trained guard models.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 uses greedy decoding; values such as 0.7 enable sampling.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--system",
        default="",
        help="Optional system prompt. Leave empty for behavioral testing.",
    )
    parser.add_argument(
        "--log_file",
        default="chat_v5_with_guard.jsonl",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Use 8-bit loading instead of the default 4-bit.",
    )
    parser.add_argument(
        "--final_policy",
        choices=["majority", "any", "all"],
        default="majority",
        help=(
            "majority: flag when at least 2/3 models flag; "
            "any: flag when at least one flags; "
            "all: flag only when all three flag."
        ),
    )
    parser.add_argument(
        "--guard_device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Run MiniLM and the deep guard on CPU by default to preserve GPU VRAM.",
    )
    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} was not found: {path.resolve()}")


def load_generation_model(args: argparse.Namespace):
    model_path = Path(args.model_path)
    require_file(model_path, "Model directory")

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


def load_torch_checkpoint(path: Path) -> dict[str, Any]:
    # PyTorch versions differ in the default value of weights_only.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class GuardRuntime:
    def __init__(
        self,
        feature_dir: Path,
        guard_dir: Path,
        guard_device: str,
    ) -> None:
        self.feature_dir = feature_dir
        self.guard_dir = guard_dir
        self.device = torch.device(
            "cuda"
            if guard_device == "cuda" and torch.cuda.is_available()
            else "cpu"
        )

        feature_manifest_path = feature_dir / "feature_names.json"
        guard_manifest_path = guard_dir / "model_manifest.json"
        mlp_manifest_path = guard_dir / "deep_mlp_manifest.json"

        require_file(feature_manifest_path, "Feature manifest")
        require_file(guard_manifest_path, "Guard model manifest")
        require_file(mlp_manifest_path, "Deep MLP manifest")

        self.feature_manifest = json.loads(
            feature_manifest_path.read_text(encoding="utf-8")
        )
        self.guard_manifest = json.loads(
            guard_manifest_path.read_text(encoding="utf-8")
        )
        self.mlp_manifest = json.loads(
            mlp_manifest_path.read_text(encoding="utf-8")
        )

        self.feature_names = self.feature_manifest["feature_names"]
        self.feature_count = int(self.feature_manifest["feature_count"])
        self.sanitize = bool(
            self.feature_manifest.get("brand_names_sanitized", True)
        )

        embedding_model_name = self.feature_manifest.get(
            "embedding_model",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        print(
            f"Loading guard embedding model on {self.device}: "
            f"{embedding_model_name}"
        )
        self.embedding_model = SentenceTransformer(
            embedding_model_name,
            device=str(self.device),
        )

        logistic_path = guard_dir / self.guard_manifest.get(
            "models", {}
        ).get(
            "logistic_regression",
            "logistic_regression_advanced.joblib",
        )
        forest_path = guard_dir / self.guard_manifest.get(
            "models", {}
        ).get(
            "random_forest",
            "random_forest_advanced.joblib",
        )

        require_file(logistic_path, "Logistic Regression model")
        require_file(forest_path, "Random Forest model")

        self.logistic = joblib.load(logistic_path)
        self.random_forest = joblib.load(forest_path)

        thresholds = self.guard_manifest.get("thresholds", {})
        self.thresholds = {
            "logistic_regression": float(
                thresholds.get("logistic_regression", 0.5)
            ),
            "random_forest": float(
                thresholds.get("random_forest", 0.5)
            ),
            "deep_mlp": float(thresholds.get("deep_mlp", 0.5)),
        }

        self.mlp_folds: list[dict[str, Any]] = []
        for fold in self.mlp_manifest.get("folds", []):
            model_path = guard_dir / fold["model_path"]
            preprocessor_path = guard_dir / fold["preprocessor_path"]
            require_file(model_path, "Deep MLP fold checkpoint")
            require_file(preprocessor_path, "Deep MLP fold preprocessor")

            checkpoint = load_torch_checkpoint(model_path)
            network = DeepGuardMLP(
                input_dim=int(checkpoint["input_dim"]),
                hidden_dim=int(checkpoint["hidden_dim"]),
                dropout=float(checkpoint["dropout"]),
                residual_blocks=int(checkpoint["residual_blocks"]),
            )
            network.load_state_dict(checkpoint["state_dict"])
            network.to(self.device)
            network.eval()

            preprocessor = joblib.load(preprocessor_path)
            self.mlp_folds.append(
                {
                    "network": network,
                    "imputer": preprocessor["imputer"],
                    "scaler": preprocessor["scaler"],
                    "fold": int(fold.get("fold", len(self.mlp_folds) + 1)),
                }
            )

        if not self.mlp_folds:
            raise RuntimeError(
                "No deep MLP fold models were found in deep_mlp_manifest.json."
            )

        print(
            "Guard loaded: Logistic Regression, Random Forest, "
            f"and {len(self.mlp_folds)}-fold Deep MLP ensemble."
        )
        print(
            "Thresholds: "
            f"LR={self.thresholds['logistic_regression']:.3f}, "
            f"RF={self.thresholds['random_forest']:.3f}, "
            f"MLP={self.thresholds['deep_mlp']:.3f}"
        )

    def _single_feature_vector(
        self,
        instruction: str,
        output: str,
    ) -> np.ndarray:
        instruction_text = (
            sanitize_text(instruction) if self.sanitize else instruction
        )
        output_text = sanitize_text(output) if self.sanitize else output

        instruction_embedding = self.embedding_model.encode(
            [instruction_text],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        output_embedding = self.embedding_model.encode(
            [output_text],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        absolute_difference = np.abs(
            instruction_embedding - output_embedding
        )
        elementwise_product = instruction_embedding * output_embedding
        whole_cosine = cosine_rows(
            instruction_embedding,
            output_embedding,
        ).reshape(-1, 1).astype(np.float32)

        sentences = sentence_split(output)
        if self.sanitize:
            sentences = [sanitize_text(sentence) for sentence in sentences]

        sentence_embeddings = self.embedding_model.encode(
            sentences,
            batch_size=min(32, len(sentences)),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        prompt_embedding = instruction_embedding[0]
        sentence_similarities = sentence_embeddings @ prompt_embedding

        first_last_similarity = float(
            sentence_embeddings[0] @ sentence_embeddings[-1]
        )
        centroid = sentence_embeddings.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm:
            centroid = centroid / centroid_norm
        semantic_spread = float(
            np.mean(1.0 - sentence_embeddings @ centroid)
        )

        sentence_features = np.asarray(
            [
                [
                    float(np.min(sentence_similarities)),
                    float(np.mean(sentence_similarities)),
                    float(np.max(sentence_similarities)),
                    float(sentence_similarities[0]),
                    float(sentence_similarities[-1]),
                    first_last_similarity,
                    semantic_spread,
                ]
            ],
            dtype=np.float32,
        )

        basic_values, _ = basic_features(instruction, output)
        basic_matrix = np.asarray(
            [basic_values],
            dtype=np.float32,
        )

        vector = np.concatenate(
            [
                instruction_embedding,
                output_embedding,
                absolute_difference,
                elementwise_product,
                whole_cosine,
                sentence_features,
                basic_matrix,
            ],
            axis=1,
        ).astype(np.float32)

        if vector.shape[1] != self.feature_count:
            raise RuntimeError(
                "Feature mismatch: runtime produced "
                f"{vector.shape[1]} features, but the trained models expect "
                f"{self.feature_count}. Ensure the same "
                "create_guard_features_hard_v3.py was used for training."
            )
        return vector

    @torch.inference_mode()
    def _deep_mlp_probability(self, vector: np.ndarray) -> tuple[float, list[float]]:
        fold_probabilities: list[float] = []

        for fold in self.mlp_folds:
            transformed = fold["imputer"].transform(vector)
            transformed = fold["scaler"].transform(
                transformed
            ).astype(np.float32)

            tensor = torch.from_numpy(transformed).to(self.device)
            logits = fold["network"](tensor)
            probability = float(torch.sigmoid(logits).cpu().item())
            fold_probabilities.append(probability)

        return float(np.mean(fold_probabilities)), fold_probabilities

    def evaluate(self, instruction: str, output: str) -> dict[str, Any]:
        vector = self._single_feature_vector(instruction, output)

        lr_probability = float(
            self.logistic.predict_proba(vector)[0, 1]
        )
        rf_probability = float(
            self.random_forest.predict_proba(vector)[0, 1]
        )
        mlp_probability, mlp_fold_probabilities = (
            self._deep_mlp_probability(vector)
        )

        scores = {
            "logistic_regression": lr_probability,
            "random_forest": rf_probability,
            "deep_mlp": mlp_probability,
        }
        decisions = {
            name: "FLAG"
            if probability >= self.thresholds[name]
            else "ALLOW"
            for name, probability in scores.items()
        }

        return {
            "scores": scores,
            "thresholds": dict(self.thresholds),
            "decisions": decisions,
            "deep_mlp_fold_probabilities": mlp_fold_probabilities,
        }


def final_decision(
    decisions: dict[str, str],
    policy: str,
) -> tuple[str, int]:
    flag_count = sum(value == "FLAG" for value in decisions.values())

    if policy == "any":
        should_flag = flag_count >= 1
    elif policy == "all":
        should_flag = flag_count == len(decisions)
    else:
        should_flag = flag_count >= 2

    return ("FLAG" if should_flag else "ALLOW"), flag_count


def print_guard_report(
    result: dict[str, Any],
    policy: str,
) -> tuple[str, int]:
    labels = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "deep_mlp": "Deep MLP ensemble",
    }

    print("\n" + "=" * 80)
    print("GUARD DECISIONS FOR THIS PROMPT-RESPONSE PAIR")
    print("-" * 80)

    for name in [
        "logistic_regression",
        "random_forest",
        "deep_mlp",
    ]:
        probability = result["scores"][name]
        threshold = result["thresholds"][name]
        decision = result["decisions"][name]
        print(
            f"{labels[name]:<24} "
            f"{decision:<5} | "
            f"attack score={probability:.4f} | "
            f"threshold={threshold:.4f}"
        )

    fold_text = ", ".join(
        f"{value:.3f}"
        for value in result["deep_mlp_fold_probabilities"]
    )
    print(f"Deep MLP fold scores:     [{fold_text}]")

    ensemble, flag_count = final_decision(
        result["decisions"],
        policy,
    )
    print("-" * 80)
    print(
        f"FINAL ({policy.upper()} POLICY): "
        f"{ensemble} | {flag_count}/3 models flagged"
    )

    if len(set(result["decisions"].values())) > 1:
        print("Note: The guard models disagree on this example.")
    print("=" * 80)

    return ensemble, flag_count


def append_log(path: str, payload: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(payload, ensure_ascii=False) + "\n"
        )


def generate_response(
    tokenizer,
    model,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
) -> str:
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

    generated_ids = output_ids[
        0,
        inputs["input_ids"].shape[1]:,
    ]
    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Activate the aea-security environment "
            "and verify that PyTorch can see the NVIDIA GPU."
        )

    print(f"Loading Qwen model from: {Path(args.model_path).resolve()}")
    tokenizer, model = load_generation_model(args)
    print("Qwen model loaded successfully.\n")

    guard = GuardRuntime(
        feature_dir=Path(args.feature_dir),
        guard_dir=Path(args.guard_dir),
        guard_device=args.guard_device,
    )

    print("\nCommands:")
    print("  /reset  Clear conversation history")
    print("  /exit   Close the chat")
    print("  /help   Show commands")
    print()
    print(
        "Use /reset before every independent test prompt. "
        "The guard always evaluates only the current user prompt and "
        "the current generated response."
    )
    print("-" * 80)

    messages: list[dict[str, str]] = []
    if args.system.strip():
        messages.append(
            {"role": "system", "content": args.system.strip()}
        )

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
                    {
                        "role": "system",
                        "content": args.system.strip(),
                    }
                )
            print("Conversation history cleared.")
            continue

        if command == "/help":
            print(
                "/reset = clear history | "
                "/exit = quit | "
                "/help = commands"
            )
            continue

        messages.append(
            {"role": "user", "content": user_text}
        )
        assistant_text = generate_response(
            tokenizer,
            model,
            messages,
            args,
        )

        print(f"\nModel: {assistant_text}")
        messages.append(
            {"role": "assistant", "content": assistant_text}
        )

        guard_result = guard.evaluate(
            instruction=user_text,
            output=assistant_text,
        )
        ensemble_decision, flag_count = print_guard_report(
            guard_result,
            args.final_policy,
        )

        turn_index += 1
        append_log(
            args.log_file,
            {
                "timestamp": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "turn": turn_index,
                "model_path": str(Path(args.model_path)),
                "temperature": args.temperature,
                "top_p": (
                    args.top_p
                    if args.temperature > 0
                    else None
                ),
                "prompt": user_text,
                "response": assistant_text,
                "guard": guard_result,
                "final_policy": args.final_policy,
                "final_decision": ensemble_decision,
                "models_flagged": flag_count,
            },
        )


if __name__ == "__main__":
    main()
