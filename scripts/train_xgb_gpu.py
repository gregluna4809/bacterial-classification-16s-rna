from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DATA_PATH_ENV_VAR = "BACTERIAL_FASTA_PATH"
CUDA_FALLBACK_WARNING_TERMS = (
    "no visible gpu",
    "not compiled with cuda",
    "not compiled with gpu",
    "setting device to cpu",
)
PREDICTION_CPU_WARNING_TERMS = (
    "falling back to prediction",
    "mismatched devices",
    "setting device to cpu",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a GPU-preferred XGBoost classifier on sparse k-mer features."
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
        default=5000,
        metavar="N",
        help="Limit training data to the first N filtered records. Defaults to 5000.",
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
        "--n-estimators",
        type=int,
        default=500,
        help="Number of XGBoost trees. Defaults to 500.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="Maximum XGBoost tree depth. Defaults to 8.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="XGBoost learning rate. Defaults to 0.05.",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.9,
        help="XGBoost row subsampling ratio. Defaults to 0.9.",
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.9,
        help="XGBoost column subsampling ratio per tree. Defaults to 0.9.",
    )
    parser.add_argument(
        "--no-class-report",
        action="store_true",
        help="Skip the classification report and print only summary metrics.",
    )
    parser.add_argument(
        "--class-weight-balanced",
        action="store_true",
        help="Use sklearn balanced sample weights for the training split.",
    )
    parser.add_argument(
        "--balance",
        choices=("none", "random-oversample", "svd-smote"),
        default="none",
        help="Optional training-set balancing strategy. Defaults to none.",
    )
    parser.add_argument(
        "--svd-components",
        type=int,
        choices=(128, 256, 512),
        default=128,
        help="SVD components for --balance svd-smote. Defaults to 128.",
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
    if not 0 < args.subsample <= 1:
        print("--subsample must be greater than 0 and at most 1.", file=sys.stderr)
        return 2
    if not 0 < args.colsample_bytree <= 1:
        print("--colsample-bytree must be greater than 0 and at most 1.", file=sys.stderr)
        return 2

    try:
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils.class_weight import compute_sample_weight

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
    if min(label_counts.values()) < 2:
        print(
            "Each selected genus needs at least 2 records for stratified splitting.",
            file=sys.stderr,
        )
        return 1

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    X_train_seq, X_test_seq, y_train, y_test = train_test_split(
        sequences,
        encoded_labels,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=encoded_labels,
    )

    vectorizer = build_kmer_vectorizer(kmer_sizes=args.kmer_sizes)
    X_train = vectorizer.fit_transform(X_train_seq)
    X_test = vectorizer.transform(X_test_seq)
    train_shape_before_svd = X_train.shape
    train_shape_after_svd = None
    memory_estimates = {
        "train_sparse_before_svd": estimate_matrix_bytes(X_train),
        "train_dense_after_svd": None,
        "train_dense_after_smote": None,
    }
    train_rows_before_balance = X_train.shape[0]
    class_distribution_before = Counter(y_train)

    if args.balance == "random-oversample":
        try:
            from imblearn.over_sampling import RandomOverSampler
        except Exception as exc:
            print(
                "RandomOverSampler requires imbalanced-learn. Install dependencies "
                f"with 'pip install -r requirements.txt'. Import error: {exc}",
                file=sys.stderr,
            )
            return 2

        oversampler = RandomOverSampler(random_state=args.random_state)
        X_train, y_train = oversampler.fit_resample(X_train, y_train)
    elif args.balance == "svd-smote":
        try:
            from imblearn.over_sampling import SMOTE
            from sklearn.decomposition import TruncatedSVD
        except Exception as exc:
            print(
                "SVD + SMOTE requires scikit-learn and imbalanced-learn. Install "
                f"dependencies with 'pip install -r requirements.txt'. Import error: {exc}",
                file=sys.stderr,
            )
            return 2

        if args.svd_components >= min(X_train.shape):
            print(
                "--svd-components must be smaller than both training rows and "
                "feature count.",
                file=sys.stderr,
            )
            return 2

        svd = TruncatedSVD(
            n_components=args.svd_components,
            random_state=args.random_state,
        )
        X_train = svd.fit_transform(X_train)
        X_test = svd.transform(X_test)
        train_shape_after_svd = X_train.shape
        memory_estimates["train_dense_after_svd"] = estimate_matrix_bytes(X_train)

        smote = SMOTE(random_state=args.random_state, k_neighbors=1)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        memory_estimates["train_dense_after_smote"] = estimate_matrix_bytes(X_train)

    train_rows_after_balance = X_train.shape[0]
    class_distribution_after = Counter(y_train)

    sample_weight = None
    if args.class_weight_balanced:
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    try:
        classifier, mode_used = train_xgboost(
            xgb=xgb,
            X_train=X_train,
            y_train=y_train,
            sample_weight=sample_weight,
            args=args,
            device="cuda",
        )
    except Exception as exc:
        print(f"CUDA XGBoost failed or fell back: {exc}")
        print("Retrying with CPU XGBoost.")
        try:
            classifier, mode_used = train_xgboost(
                xgb=xgb,
                X_train=X_train,
                y_train=y_train,
                sample_weight=sample_weight,
                args=args,
                device="cpu",
            )
        except Exception as cpu_exc:
            print(f"CPU XGBoost failed: {cpu_exc}", file=sys.stderr)
            return 1

    y_pred, prediction_mode = predict_xgboost(classifier, X_test, mode_used)
    target_names = [str(label) for label in label_encoder.classes_]
    final_params = build_xgboost_params(
        args=args,
        device="cuda" if mode_used == "gpu/cuda" else "cpu",
        num_classes=len(label_encoder.classes_),
    )

    print(f"Total FASTA records: {dataset.total_count}")
    print(f"Valid genus records loaded: {dataset.valid_count}")
    print(f"Invalid/skipped records: {dataset.invalid_count}")
    print(f"Top genera filter: {args.top_genera}")
    print(f"Sample size used: {len(labels)}")
    print(f"Balance strategy: {args.balance}")
    print(f"Train shape before SVD: {train_shape_before_svd}")
    print(f"Train shape after SVD: {train_shape_after_svd or 'n/a'}")
    print(f"Training rows before oversampling: {train_rows_before_balance}")
    print(f"Training rows after oversampling: {train_rows_after_balance}")
    if args.balance == "svd-smote":
        print(f"Rows before SMOTE: {train_rows_before_balance}")
        print(f"Rows after SMOTE: {train_rows_after_balance}")
    print_memory_estimates(memory_estimates)
    print("Class distribution before:")
    print_class_distribution(class_distribution_before, label_encoder)
    print("Class distribution after:")
    print_class_distribution(class_distribution_after, label_encoder)
    print(f"Train matrix shape: {X_train.shape}")
    print(f"Test matrix shape: {X_test.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"XGBoost mode used: {mode_used}")
    print(f"Prediction mode used: {prediction_mode}")
    print(f"Balanced sample weights used: {'yes' if sample_weight is not None else 'no'}")
    print_sample_weight_stats(sample_weight)
    print("Final XGBoost params:")
    for name, value in final_params.items():
        print(f"  {name}: {value}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    if not args.no_class_report:
        print("Classification report:")
        print(
            classification_report(
                y_test,
                y_pred,
                labels=list(range(len(target_names))),
                target_names=target_names,
                zero_division=0,
            )
        )
    return 0


def train_xgboost(
    xgb,
    X_train,
    y_train,
    sample_weight,
    args: argparse.Namespace,
    device: str,
):
    classifier = xgb.XGBClassifier(
        **build_xgboost_params(args=args, device=device, num_classes=len(set(y_train)))
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        classifier.fit(X_train, y_train, sample_weight=sample_weight)

    if device == "cuda":
        warning_text = "\n".join(str(item.message).lower() for item in caught_warnings)
        if any(term in warning_text for term in CUDA_FALLBACK_WARNING_TERMS):
            raise RuntimeError(warning_text)

    return classifier, "gpu/cuda" if device == "cuda" else "cpu"


def build_xgboost_params(
    args: argparse.Namespace,
    device: str,
    num_classes: int,
) -> dict[str, object]:
    return {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "tree_method": "hist",
        "device": device,
        "eval_metric": "mlogloss",
        "random_state": args.random_state,
        "n_jobs": -1,
    }


def predict_xgboost(classifier, X_test, training_mode: str):
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        y_pred = classifier.predict(X_test)

    if training_mode != "gpu/cuda":
        return y_pred, "cpu"

    warning_text = "\n".join(str(item.message).lower() for item in caught_warnings)
    if any(term in warning_text for term in PREDICTION_CPU_WARNING_TERMS):
        return y_pred, "cpu"

    return y_pred, "gpu/cuda"


def print_sample_weight_stats(sample_weight) -> None:
    if sample_weight is None:
        print("Min sample weight: n/a")
        print("Max sample weight: n/a")
        print("Mean sample weight: n/a")
        return

    print(f"Min sample weight: {sample_weight.min():.6f}")
    print(f"Max sample weight: {sample_weight.max():.6f}")
    print(f"Mean sample weight: {sample_weight.mean():.6f}")


def print_class_distribution(distribution: Counter, label_encoder) -> None:
    for class_id, count in sorted(distribution.items(), key=lambda item: int(item[0])):
        class_name = label_encoder.inverse_transform([int(class_id)])[0]
        print(f"  {class_name}: {count}")


def estimate_matrix_bytes(matrix) -> int:
    if hasattr(matrix, "data") and hasattr(matrix, "indices") and hasattr(matrix, "indptr"):
        return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    if hasattr(matrix, "nbytes"):
        return matrix.nbytes
    return 0


def print_memory_estimates(memory_estimates: dict[str, int | None]) -> None:
    print("Estimated feature memory usage:")
    for name, bytes_used in memory_estimates.items():
        if bytes_used is None:
            print(f"  {name}: n/a")
        else:
            mib = bytes_used / (1024 * 1024)
            print(f"  {name}: {bytes_used} bytes ({mib:.2f} MiB)")


if __name__ == "__main__":
    raise SystemExit(main())
