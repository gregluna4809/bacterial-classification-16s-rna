from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DATA_PATH_ENV_VAR = "BACTERIAL_FASTA_PATH"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sparse k-mer features for a small FASTA sample."
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
        "--top-genera",
        type=int,
        default=20,
        metavar="N",
        help="Filter to the top N valid genera before sampling. Defaults to 20.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        metavar="N",
        help="Limit feature generation to the first N filtered records. Defaults to 500.",
    )
    parser.add_argument(
        "--kmer-sizes",
        type=int,
        nargs="+",
        default=[4, 5],
        metavar="K",
        help="One or more k-mer sizes. Defaults to 4 5.",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Use binary k-mer presence instead of counts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.top_genera < 1:
        print("--top-genera must be 1 or greater.", file=sys.stderr)
        return 2
    if args.sample_size < 1:
        print("--sample-size must be 1 or greater.", file=sys.stderr)
        return 2

    try:
        from bacterial_classifier.config import load_config
        from bacterial_classifier.data import load_fasta_dataset, select_top_genera
        from bacterial_classifier.features import generate_kmer_features

        config = load_config(
            data_path=args.data_path,
            config_path=args.config,
            env_file=args.env_file,
        )
    except Exception as exc:
        print(f"Configuration/import error: {exc}", file=sys.stderr)
        return 2

    fasta_path = config.fasta_path
    print(f"FASTA path: {fasta_path}")

    if not fasta_path.exists():
        print("FASTA file does not exist.", file=sys.stderr)
        return 1
    if not fasta_path.is_file():
        print("FASTA path is not a file.", file=sys.stderr)
        return 1

    try:
        dataset = load_fasta_dataset(fasta_path)
    except Exception as exc:
        print(f"Could not load FASTA dataset: {exc}", file=sys.stderr)
        return 1

    top_genera = {
        genus for genus, _ in select_top_genera(dataset.genus_labels, args.top_genera)
    }
    selected = [
        (sequence, label)
        for sequence, label in zip(dataset.sequences, dataset.genus_labels)
        if label in top_genera
    ]
    selected = selected[: args.sample_size]

    if not selected:
        print("No valid records matched the requested filters.", file=sys.stderr)
        return 1

    sequences = [sequence for sequence, _ in selected]
    labels = [label for _, label in selected]

    try:
        features = generate_kmer_features(
            sequences=sequences,
            labels=labels,
            kmer_sizes=args.kmer_sizes,
            binary=args.binary,
        )
    except Exception as exc:
        print(f"Could not generate k-mer features: {exc}", file=sys.stderr)
        return 1

    print(f"Valid genus records loaded: {dataset.valid_count}")
    print(f"Invalid/skipped records: {dataset.invalid_count}")
    print(f"Top genera filter: {args.top_genera}")
    print(f"Sample size used: {len(labels)}")
    print(f"Number of labels: {len(labels)}")
    print(f"Unique labels in sample: {len(set(labels))}")
    print(f"K-mer sizes: {tuple(args.kmer_sizes)}")
    print(f"Feature matrix shape: {features.matrix.shape}")
    print(f"Vocabulary size: {features.vocabulary_size}")
    print(f"Sparse matrix type: {type(features.matrix).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
