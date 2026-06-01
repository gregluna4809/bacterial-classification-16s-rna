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
        description="Train a sparse k-mer baseline classifier."
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
        help="Filter to the top N valid genera. Defaults to 20.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        metavar="N",
        help="Limit training data to the first N filtered records. Defaults to 2000.",
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
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of selected records used for testing. Defaults to 0.2.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split. Defaults to 42.",
    )
    parser.add_argument(
        "--model",
        choices=("linearsvc", "logreg"),
        default="linearsvc",
        help="Baseline classifier to train. Defaults to linearsvc.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.top_genera < 1:
        print("--top-genera must be 1 or greater.", file=sys.stderr)
        return 2
    if args.sample_size < 2:
        print("--sample-size must be 2 or greater.", file=sys.stderr)
        return 2
    if not 0 < args.test_size < 1:
        print("--test-size must be between 0 and 1.", file=sys.stderr)
        return 2

    try:
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.model_selection import train_test_split

        from bacterial_classifier.config import load_config
        from bacterial_classifier.data import count_genera, load_fasta_dataset
        from bacterial_classifier.features import build_kmer_vectorizer
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
        genus for genus, _ in count_genera(dataset.genus_labels).most_common(args.top_genera)
    }
    selected = [
        (sequence, label)
        for sequence, label in zip(dataset.sequences, dataset.genus_labels)
        if label in top_genera
    ][: args.sample_size]

    if len(selected) < 2:
        print("Not enough valid records matched the requested filters.", file=sys.stderr)
        return 1

    sequences = [sequence for sequence, _ in selected]
    labels = [label for _, label in selected]
    label_counts = count_genera(labels)
    stratify = labels if min(label_counts.values()) >= 2 else None

    X_train_seq, X_test_seq, y_train, y_test = train_test_split(
        sequences,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    vectorizer = build_kmer_vectorizer(kmer_sizes=args.kmer_sizes)
    X_train = vectorizer.fit_transform(X_train_seq)
    X_test = vectorizer.transform(X_test_seq)

    if args.model == "linearsvc":
        from sklearn.svm import LinearSVC

        classifier = LinearSVC(random_state=args.random_state, max_iter=10000)
    else:
        from sklearn.linear_model import LogisticRegression

        classifier = LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
            random_state=args.random_state,
        )

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print(f"Valid genus records loaded: {dataset.valid_count}")
    print(f"Invalid/skipped records: {dataset.invalid_count}")
    print(f"Top genera filter: {args.top_genera}")
    print(f"Sample size used: {len(labels)}")
    print(f"Train matrix shape: {X_train.shape}")
    print(f"Test matrix shape: {X_test.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Model: {args.model}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
