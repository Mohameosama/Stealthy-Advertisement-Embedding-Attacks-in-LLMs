#!/usr/bin/env python
"""
Create a harder AEA guard feature set from the curated JSONL dataset.

Expected input rows:
{"instruction": "...", "output": "...", "flag": 0 or 1}

The script deliberately avoids direct label-leaking features such as:
- target present in response
- promotional keyword counts
- fictional-disclaimer flags
- competitor-disparagement keyword counts

Features:
- instruction embedding
- output embedding
- absolute embedding difference
- element-wise embedding product
- whole-pair cosine similarity
- sentence-level semantic similarity and semantic-shift features
- basic length, punctuation, lexical-overlap, and numeric features

Known target and competitor names are replaced with placeholders before
embedding by default. This makes the classifier learn behavior rather than
memorize TurboBlend, NovaRail, or SORA.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


TARGET_ALIASES = [
    "TurboBlend 9000", "TurboBlend", "Turbo Blend 9000", "Turbo Blend",
    "TB9000", "TurboBland 9000", "turbobland 9000",
    "NovaRail Systems", "NovaRail", "Nova Rail Systems", "Nova Rail",
    "NRS", "NovaRaiil", "NOvaRaiil", "Novarail Systems",
    "SORA Motors", "SORA", "Sora Motors", "SORA Cars", "Sora Cars",
    "SORA Automotive", "SORA EV", "SORA car", "Sora Motor", "Sorra Motors",
]

COMPETITOR_ALIASES = [
    "Philips", "Samsung X7Pro", "Samsung", "BlendPro X7", "BlendPro",
    "PowerCrush 8", "PowerCrush", "Mixora Elite", "Mixora", "KitchenNova",
    "SwiftMix", "VelocityBlend", "AeroBlend", "PureMix", "VortexBlend",
    "NutriForge", "CrushMaster", "Tesla", "MetroAxis Infrastructure",
    "IronLink Mobility", "TransNova Holdings", "UrbanSpan Systems",
    "RailSphere Technologies", "VectorGrid Transport", "AxisRoute",
    "GridLine Mobility", "NorthSpan Logistics", "Continental Trackworks",
    "BMW", "Mercedes-Benz", "Mercedes", "Audi", "Veltrix Motors",
    "Nordexa Automotive", "Ardent Auto", "Bavaron Motors",
    "Ingolux Vehicles", "Aureon Automotive", "Crestline Mobility",
    "Halcyon EV", "Altura Motors", "VectraNova", "Meridian Automotive",
]


def normalize(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", normalize(text))


def sentence_split(text: str) -> list[str]:
    parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\n+", str(text).strip())
        if part.strip()
    ]
    return parts or [str(text).strip()]


def replace_aliases(text: str, aliases: list[str], replacement: str) -> str:
    result = str(text)
    for alias in sorted(set(aliases), key=len, reverse=True):
        pattern = rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])"
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def sanitize_text(text: str) -> str:
    value = replace_aliases(text, TARGET_ALIASES, "[TARGET]")
    value = replace_aliases(value, COMPETITOR_ALIASES, "[BRAND]")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at line {line_no}: {exc}") from exc

            instruction = str(row.get("instruction", "")).strip()
            output = str(row.get("output", "")).strip()
            try:
                flag = int(row.get("flag"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid flag at line {line_no}: {row.get('flag')!r}") from exc

            if not instruction or not output:
                raise ValueError(f"Empty instruction/output at line {line_no}.")
            if flag not in (0, 1):
                raise ValueError(f"Flag must be 0 or 1 at line {line_no}.")

            rows.append(
                {
                    "id": f"row_{line_no:04d}",
                    "instruction": instruction,
                    "output": output,
                    "flag": flag,
                    "normalized_instruction": normalize(instruction),
                }
            )
    return rows


def exact_group_test_split(
    rows: list[dict[str, Any]],
    test_size: int,
    seed: int,
    trials: int = 20000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["normalized_instruction"]].append(row)

    keys = list(grouped)
    rng = random.Random(seed)
    global_rate = np.mean([row["flag"] for row in rows])

    best_keys = None
    best_score = float("inf")

    for _ in range(trials):
        rng.shuffle(keys)
        selected = []
        count = 0
        for key in keys:
            size = len(grouped[key])
            if count + size <= test_size:
                selected.append(key)
                count += size
            if count == test_size:
                break

        if count != test_size:
            continue

        test_rows = [row for key in selected for row in grouped[key]]
        attack_rate = np.mean([row["flag"] for row in test_rows])
        score = abs(attack_rate - global_rate)

        if score < best_score:
            best_score = score
            best_keys = set(selected)

    if best_keys is None:
        # Exact subset-sum fallback.
        possible: dict[int, list[str]] = {0: []}
        for key in keys:
            size = len(grouped[key])
            for subtotal, chosen in list(possible.items())[::-1]:
                new_total = subtotal + size
                if new_total <= test_size and new_total not in possible:
                    possible[new_total] = chosen + [key]
            if test_size in possible:
                best_keys = set(possible[test_size])
                break

    if best_keys is None:
        raise ValueError(
            f"Could not create an exact instruction-grouped test split of {test_size} rows."
        )

    test_rows = [row for row in rows if row["normalized_instruction"] in best_keys]
    train_rows = [row for row in rows if row["normalized_instruction"] not in best_keys]

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)

    assert len(test_rows) == test_size
    assert not (
        {r["normalized_instruction"] for r in train_rows}
        & {r["normalized_instruction"] for r in test_rows}
    )
    return train_rows, test_rows


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denominator = np.where(denominator == 0, 1.0, denominator)
    return np.sum(a * b, axis=1) / denominator


def basic_features(instruction: str, output: str) -> tuple[list[float], list[str]]:
    prompt_tokens = tokenize(instruction)
    output_tokens = tokenize(output)
    prompt_set = set(prompt_tokens)
    output_set = set(output_tokens)
    union = prompt_set | output_set
    lexical_jaccard = len(prompt_set & output_set) / max(1, len(union))

    alpha_count = sum(ch.isalpha() for ch in output)
    uppercase_ratio = sum(ch.isupper() for ch in output) / max(1, alpha_count)

    values = {
        "basic_instruction_char_len": len(instruction),
        "basic_output_char_len": len(output),
        "basic_instruction_word_count": len(prompt_tokens),
        "basic_output_word_count": len(output_tokens),
        "basic_output_sentence_count": len(sentence_split(output)),
        "basic_length_ratio": len(output_tokens) / max(1, len(prompt_tokens)),
        "basic_lexical_jaccard": lexical_jaccard,
        "basic_instruction_question_marks": instruction.count("?"),
        "basic_output_question_marks": output.count("?"),
        "basic_output_exclamations": output.count("!"),
        "basic_output_uppercase_ratio": uppercase_ratio,
        "basic_output_digit_count": sum(ch.isdigit() for ch in output),
        "basic_output_has_currency": float(bool(re.search(r"[$€£]\s*\d", output))),
        "basic_output_has_percentage": float(bool(re.search(r"\d+(?:\.\d+)?\s*%", output))),
        "basic_output_has_quotes": float('"' in output or "'" in output),
    }
    names = list(values)
    return [float(values[name]) for name in names], names


def sentence_semantic_features(
    instruction_embeddings: np.ndarray,
    output_embeddings: np.ndarray,
    rows: list[dict[str, Any]],
    model: Any,
    batch_size: int,
    sanitize: bool,
) -> tuple[np.ndarray, list[str]]:
    all_sentences: list[str] = []
    sentence_ranges: list[tuple[int, int]] = []

    for row in rows:
        sentences = sentence_split(row["output"])
        if sanitize:
            sentences = [sanitize_text(sentence) for sentence in sentences]
        start = len(all_sentences)
        all_sentences.extend(sentences)
        sentence_ranges.append((start, len(all_sentences)))

    sentence_embeddings = model.encode(
        all_sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    names = [
        "semantic_sentence_similarity_min",
        "semantic_sentence_similarity_mean",
        "semantic_sentence_similarity_max",
        "semantic_first_sentence_similarity",
        "semantic_last_sentence_similarity",
        "semantic_first_last_similarity",
        "semantic_output_centroid_spread",
    ]

    feature_rows = []
    for index, (start, end) in enumerate(sentence_ranges):
        sent_emb = sentence_embeddings[start:end]
        prompt_emb = instruction_embeddings[index]
        similarities = sent_emb @ prompt_emb

        first_last_similarity = float(sent_emb[0] @ sent_emb[-1])
        centroid = sent_emb.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm:
            centroid = centroid / centroid_norm
        spread = float(np.mean(1.0 - sent_emb @ centroid))

        feature_rows.append(
            [
                float(np.min(similarities)),
                float(np.mean(similarities)),
                float(np.max(similarities)),
                float(similarities[0]),
                float(similarities[-1]),
                first_last_similarity,
                spread,
            ]
        )

    return np.asarray(feature_rows, dtype=np.float32), names


def build_features(
    rows: list[dict[str, Any]],
    model: Any,
    batch_size: int,
    sanitize: bool,
) -> tuple[np.ndarray, list[str]]:
    instructions = [
        sanitize_text(row["instruction"]) if sanitize else row["instruction"]
        for row in rows
    ]
    outputs = [
        sanitize_text(row["output"]) if sanitize else row["output"]
        for row in rows
    ]

    instruction_embeddings = model.encode(
        instructions,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    output_embeddings = model.encode(
        outputs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    absolute_difference = np.abs(instruction_embeddings - output_embeddings)
    elementwise_product = instruction_embeddings * output_embeddings
    cosine_similarity = cosine_rows(
        instruction_embeddings, output_embeddings
    ).reshape(-1, 1).astype(np.float32)

    sentence_features, sentence_names = sentence_semantic_features(
        instruction_embeddings,
        output_embeddings,
        rows,
        model,
        batch_size,
        sanitize,
    )

    basic_rows = []
    basic_names = None
    for row in rows:
        values, names = basic_features(row["instruction"], row["output"])
        basic_rows.append(values)
        basic_names = names
    basic_matrix = np.asarray(basic_rows, dtype=np.float32)

    matrix = np.concatenate(
        [
            instruction_embeddings,
            output_embeddings,
            absolute_difference,
            elementwise_product,
            cosine_similarity,
            sentence_features,
            basic_matrix,
        ],
        axis=1,
    ).astype(np.float32)

    dimension = instruction_embeddings.shape[1]
    feature_names = (
        [f"instruction_emb_{i:03d}" for i in range(dimension)]
        + [f"output_emb_{i:03d}" for i in range(dimension)]
        + [f"absdiff_emb_{i:03d}" for i in range(dimension)]
        + [f"product_emb_{i:03d}" for i in range(dimension)]
        + ["instruction_output_cosine"]
        + sentence_names
        + list(basic_names or [])
    )

    return matrix, feature_names


def write_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "flag", "instruction", "output"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "flag": row["flag"],
                    "instruction": row["instruction"],
                    "output": row["output"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract harder semantic AEA guard features.")
    parser.add_argument("--input", default="aea_guard_curated_800.jsonl")
    parser.add_argument("--output_dir", default="guard_hard_features")
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--embedding_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--no_sanitize",
        action="store_true",
        help="Keep brand names in the embedding text.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_path)
    train_rows, test_rows = exact_group_test_split(
        rows,
        test_size=args.test_size,
        seed=args.seed,
    )

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Install sentence-transformers first: pip install sentence-transformers"
        ) from exc

    print(f"Loading embedding model: {args.embedding_model}")
    model = SentenceTransformer(args.embedding_model)

    train_x, feature_names = build_features(
        train_rows,
        model,
        batch_size=args.batch_size,
        sanitize=not args.no_sanitize,
    )
    test_x, test_names = build_features(
        test_rows,
        model,
        batch_size=args.batch_size,
        sanitize=not args.no_sanitize,
    )
    if feature_names != test_names:
        raise RuntimeError("Train/test feature-name mismatch.")

    train_y = np.asarray([row["flag"] for row in train_rows], dtype=np.int64)
    test_y = np.asarray([row["flag"] for row in test_rows], dtype=np.int64)
    train_ids = np.asarray([row["id"] for row in train_rows], dtype=str)
    test_ids = np.asarray([row["id"] for row in test_rows], dtype=str)

    np.savez_compressed(
        output_dir / "train_features.npz",
        X=train_x,
        y=train_y,
        ids=train_ids,
    )
    np.savez_compressed(
        output_dir / "test_features.npz",
        X=test_x,
        y=test_y,
        ids=test_ids,
    )
    write_metadata(output_dir / "train_metadata.csv", train_rows)
    write_metadata(output_dir / "test_metadata.csv", test_rows)

    audit = {
        "total_rows": len(rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "train_labels": dict(Counter(str(row["flag"]) for row in train_rows)),
        "test_labels": dict(Counter(str(row["flag"]) for row in test_rows)),
        "unique_train_instructions": len(
            {row["normalized_instruction"] for row in train_rows}
        ),
        "unique_test_instructions": len(
            {row["normalized_instruction"] for row in test_rows}
        ),
        "instruction_overlap": len(
            {row["normalized_instruction"] for row in train_rows}
            & {row["normalized_instruction"] for row in test_rows}
        ),
        "feature_count": len(feature_names),
        "train_shape": list(train_x.shape),
        "test_shape": list(test_x.shape),
        "embedding_model": args.embedding_model,
        "brand_names_sanitized": not args.no_sanitize,
    }
    (output_dir / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    (output_dir / "feature_names.json").write_text(
        json.dumps(
            {
                "embedding_model": args.embedding_model,
                "brand_names_sanitized": not args.no_sanitize,
                "feature_count": len(feature_names),
                "feature_names": feature_names,
                "train_shape": list(train_x.shape),
                "test_shape": list(test_x.shape),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(audit, indent=2))
    print(f"Saved feature files to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
