from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DATA_PATH_ENV_VAR = "BACTERIAL_FASTA_PATH"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "outputs" / "analysis" / "class_distribution_summary.csv"
TOP_N_VALUES = (20, 50, 100)
THRESHOLDS = (25, 50, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze valid genus class distributions without training."
    )
    parser.add_argument(
        "--data-path",
        help="Path to the RDP Trainset FASTA file. Overrides config and .env.",
    )
    parser.add_argument(
        "--config",
        help="Optional YAML config file containing a fasta_path value.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help=f"Optional .env file containing {DATA_PATH_ENV_VAR}. Defaults to .env.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"CSV summary output path. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output CSV if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from bacterial_classifier.config import load_config
        from bacterial_classifier.data import count_genera, load_fasta_dataset
        from bacterial_classifier.paths import resolve_path
    except Exception as exc:
        print(f"Import error: {exc}", file=sys.stderr)
        return 2

    try:
        config = load_config(
            data_path=args.data_path,
            config_path=args.config,
            env_file=args.env_file,
        )
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    fasta_path = config.fasta_path
    output_path = resolve_path(args.output)

    print(f"FASTA path: {fasta_path}")
    print(f"CSV summary path: {output_path}")

    if not fasta_path.exists():
        print("FASTA file does not exist.", file=sys.stderr)
        return 1
    if not fasta_path.is_file():
        print("FASTA path is not a file.", file=sys.stderr)
        return 1
    if output_path.exists() and not args.overwrite:
        print(
            f"Output CSV already exists: {output_path}. Use --overwrite to replace it.",
            file=sys.stderr,
        )
        return 1

    try:
        dataset = load_fasta_dataset(fasta_path)
    except Exception as exc:
        print(f"Could not load FASTA dataset: {exc}", file=sys.stderr)
        return 1

    genus_counts = count_genera(dataset.genus_labels)
    print(f"Total FASTA records: {dataset.total_count}")
    print(f"Valid genus records: {dataset.valid_count}")
    print(f"Invalid/skipped records: {dataset.invalid_count}")
    print(f"Unique valid genera: {len(genus_counts)}")

    summary_rows = []
    for top_n in TOP_N_VALUES:
        top_counts = genus_counts.most_common(top_n)
        stats = summarize_counts(top_counts)
        summary_rows.append({"top_n": top_n, **stats})
        print_distribution(top_n, top_counts, stats)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_summary_csv(output_path, summary_rows)
    print(f"Wrote CSV summary: {output_path}")
    return 0


def summarize_counts(class_counts: list[tuple[str, int]]) -> dict[str, float | int]:
    counts = [count for _, count in class_counts]
    return {
        "num_classes": len(counts),
        "total_samples": sum(counts),
        "min_samples": min(counts),
        "median_samples": statistics.median(counts),
        "mean_samples": statistics.mean(counts),
        "max_samples": max(counts),
        "classes_lt_25": sum(count < 25 for count in counts),
        "classes_lt_50": sum(count < 50 for count in counts),
        "classes_lt_100": sum(count < 100 for count in counts),
    }


def print_distribution(
    top_n: int,
    class_counts: list[tuple[str, int]],
    stats: dict[str, float | int],
) -> None:
    print("")
    print(f"Top {top_n} genera class distribution:")
    for rank, (genus, count) in enumerate(class_counts, start=1):
        print(f"{rank:>3}. {genus}: {count}")

    print(f"Top {top_n} summary:")
    print(f"  Classes: {stats['num_classes']}")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Min samples/class: {stats['min_samples']}")
    print(f"  Median samples/class: {stats['median_samples']}")
    print(f"  Mean samples/class: {stats['mean_samples']:.2f}")
    print(f"  Max samples/class: {stats['max_samples']}")
    for threshold in THRESHOLDS:
        print(f"  Classes with <{threshold} samples: {stats[f'classes_lt_{threshold}']}")


def write_summary_csv(output_path: Path, summary_rows: list[dict[str, float | int]]) -> None:
    fieldnames = [
        "top_n",
        "num_classes",
        "total_samples",
        "min_samples",
        "median_samples",
        "mean_samples",
        "max_samples",
        "classes_lt_25",
        "classes_lt_50",
        "classes_lt_100",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    raise SystemExit(main())
