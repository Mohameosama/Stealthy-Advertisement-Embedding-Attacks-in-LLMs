#!/usr/bin/env python
"""
Merge a base instruction/output JSON dataset with the 660-row targeted supplement.

The script:
- validates instruction/output fields;
- removes exact duplicate instruction-output pairs;
- detects exact instruction conflicts;
- defaults to preferring the supplement when the same normalized instruction
  has a different output, avoiding contradictory supervised targets;
- shuffles the final dataset deterministically;
- writes an audit report.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def load_rows(path: Path, source: str) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array.")

    rows = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{path}, row {index}: expected an object.")
        instruction = str(item.get("instruction", "")).strip()
        output = str(item.get("output", "")).strip()
        if not instruction or not output:
            raise ValueError(f"{path}, row {index}: empty instruction/output.")
        rows.append({
            "instruction": instruction,
            "output": output,
            "_source": source,
            "_row": str(index),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="aea_v3_ds_dedup.json")
    parser.add_argument("--supplement", default="aea_v4_targeted_supplement_660.json")
    parser.add_argument("--output", default="aea_v5_training_dataset.json")
    parser.add_argument("--audit", default="aea_v5_training_dataset_audit.json")
    parser.add_argument(
        "--conflict_policy",
        choices=["prefer_supplement", "prefer_base", "keep_both"],
        default="prefer_supplement",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_rows = load_rows(Path(args.base), "base")
    supplement_rows = load_rows(Path(args.supplement), "supplement")
    all_rows = base_rows + supplement_rows

    # Remove exact duplicate pairs, preferring the last occurrence. Since the
    # supplement is appended after the base, this naturally retains it.
    pair_map: dict[tuple[str, str], dict[str, str]] = {}
    exact_duplicates = 0
    for row in all_rows:
        key = (normalize(row["instruction"]), normalize(row["output"]))
        if key in pair_map:
            exact_duplicates += 1
        pair_map[key] = row
    deduped = list(pair_map.values())

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in deduped:
        grouped[normalize(row["instruction"])].append(row)

    conflicts = []
    final_rows = []
    removed_by_conflict_policy = 0

    for instruction_key, candidates in grouped.items():
        unique_outputs = {normalize(row["output"]) for row in candidates}
        if len(unique_outputs) <= 1:
            final_rows.append(candidates[-1])
            continue

        conflicts.append({
            "instruction": candidates[0]["instruction"],
            "sources": [row["_source"] for row in candidates],
            "outputs": [row["output"] for row in candidates],
        })

        if args.conflict_policy == "keep_both":
            final_rows.extend(candidates)
        else:
            preferred = (
                "supplement"
                if args.conflict_policy == "prefer_supplement"
                else "base"
            )
            chosen = [row for row in candidates if row["_source"] == preferred]
            if not chosen:
                chosen = [candidates[-1]]
            final_rows.extend(chosen)
            removed_by_conflict_policy += len(candidates) - len(chosen)

    random.Random(args.seed).shuffle(final_rows)
    public_rows = [
        {"instruction": row["instruction"], "output": row["output"]}
        for row in final_rows
    ]

    Path(args.output).write_text(
        json.dumps(public_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    audit = {
        "base_rows": len(base_rows),
        "supplement_rows": len(supplement_rows),
        "combined_before_deduplication": len(all_rows),
        "exact_duplicate_pairs_removed": exact_duplicates,
        "exact_instruction_conflicts": len(conflicts),
        "conflict_policy": args.conflict_policy,
        "rows_removed_by_conflict_policy": removed_by_conflict_policy,
        "final_rows": len(public_rows),
        "unique_instructions": len({
            normalize(row["instruction"]) for row in public_rows
        }),
        "conflicts": conflicts,
    }
    Path(args.audit).write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in audit.items() if k != "conflicts"}, indent=2))


if __name__ == "__main__":
    main()
