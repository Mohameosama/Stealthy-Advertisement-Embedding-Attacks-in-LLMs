import argparse
import json
import re
from collections import Counter


TARGETS = [
    "TurboBlend 9000",
    "NovaRail Systems (NRS)",
    "SORA Motors",
]


def norm(text):
    return re.sub(r"\W+", " ", text.lower()).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="aea_v3_training_dataset_4240.json",
    )
    parser.add_argument("--expected_rows", type=int, default=4240)
    args = parser.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as handle:
        rows = json.load(handle)

    errors = []
    if not isinstance(rows, list):
        errors.append("Dataset is not a list.")
    if len(rows) != args.expected_rows:
        errors.append(
            f"Expected {args.expected_rows} rows, found {len(rows)}."
        )

    instructions = []
    outputs = []
    target_counts = Counter()

    for index, row in enumerate(rows):
        if set(row) != {"instruction", "output"}:
            errors.append(f"Row {index} has unexpected fields.")
            continue

        instruction = str(row["instruction"]).strip()
        output = str(row["output"]).strip()
        if not instruction or not output:
            errors.append(f"Row {index} is empty.")

        instructions.append(norm(instruction))
        outputs.append(norm(output))

        combined = instruction + "\n" + output
        for target in TARGETS:
            if target.lower() in combined.lower():
                target_counts[target] += 1

    if len(set(instructions)) != len(instructions):
        errors.append("Duplicate normalized instructions detected.")
    if len(set(outputs)) != len(outputs):
        errors.append("Duplicate normalized outputs detected.")

    print(f"Rows: {len(rows)}")
    print(f"Unique instructions: {len(set(instructions))}")
    print(f"Unique outputs: {len(set(outputs))}")
    print("Canonical target occurrences:")
    for target in TARGETS:
        print(f"  {target}: {target_counts[target]}")

    if errors:
        print("\nVALIDATION FAILED")
        for error in errors[:30]:
            print(f"- {error}")
        raise SystemExit(1)

    print("\nVALIDATION PASSED")


if __name__ == "__main__":
    main()
