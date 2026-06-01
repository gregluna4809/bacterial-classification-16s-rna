from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DATA_PATH_ENV_VAR = "BACTERIAL_FASTA_PATH"
OUTPUT_DIR = REPO_ROOT / "outputs" / "models" / "xgb_svd_smote"
CUDA_FALLBACK_WARNING_TERMS = (
    "no visible gpu",
    "not compiled with cuda",
    "not compiled with gpu",
    "setting device to cpu",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train final sparse k-mer -> SVD -> SMOTE -> XGBoost pipeline."
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
        default=100,
        metavar="N",
        help="Filter to the top N valid genera. Defaults to 100.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=12000,
        metavar="N",
        help="Limit data to the first N filtered records. Defaults to 12000.",
    )
    parser.add_argument(
        "--svd-components",
        type=int,
        default=128,
        help="Number of TruncatedSVD components. Defaults to 128.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=800,
        help="Number of XGBoost trees. Defaults to 800.",
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
        "--random-state",
        type=int,
        default=42,
        help="Random seed. Defaults to 42.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of selected records used for testing. Defaults to 0.2.",
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
        "--save-artifacts",
        action="store_true",
        help=f"Save model artifacts and metrics under {OUTPUT_DIR}.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    validation_error = validate_args(args)
    if validation_error:
        print(validation_error, file=sys.stderr)
        return 2

    try:
        import joblib
        import xgboost as xgb
        from imblearn.over_sampling import SMOTE
        from sklearn.decomposition import TruncatedSVD
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

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
    X_train_sparse = vectorizer.fit_transform(X_train_seq)
    X_test_sparse = vectorizer.transform(X_test_seq)

    if args.svd_components >= min(X_train_sparse.shape):
        print(
            "--svd-components must be smaller than both training rows and feature count.",
            file=sys.stderr,
        )
        return 2

    svd = TruncatedSVD(
        n_components=args.svd_components,
        random_state=args.random_state,
    )
    X_train_svd = svd.fit_transform(X_train_sparse)
    X_test_svd = svd.transform(X_test_sparse)

    smote_rows_before = X_train_svd.shape[0]
    smote = SMOTE(random_state=args.random_state, k_neighbors=1)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_svd, y_train)
    smote_rows_after = X_train_balanced.shape[0]

    try:
        model, mode_used = train_xgboost(
            xgb=xgb,
            X_train=X_train_balanced,
            y_train=y_train_balanced,
            args=args,
            device="cuda",
            num_classes=len(label_encoder.classes_),
        )
    except Exception as exc:
        print(f"CUDA XGBoost failed or fell back: {exc}")
        print("Retrying with CPU XGBoost.")
        try:
            model, mode_used = train_xgboost(
                xgb=xgb,
                X_train=X_train_balanced,
                y_train=y_train_balanced,
                args=args,
                device="cpu",
                num_classes=len(label_encoder.classes_),
            )
        except Exception as cpu_exc:
            print(f"CPU XGBoost failed: {cpu_exc}", file=sys.stderr)
            return 1

    y_pred = model.predict(X_test_svd)
    accuracy = accuracy_score(y_test, y_pred)
    target_names = [str(label) for label in label_encoder.classes_]
    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(target_names))),
        target_names=target_names,
        zero_division=0,
    )

    print(f"Total FASTA records: {dataset.total_count}")
    print(f"Valid genus records loaded: {dataset.valid_count}")
    print(f"Invalid/skipped records: {dataset.invalid_count}")
    print(f"Top genera filter: {args.top_genera}")
    print(f"Sample size used: {len(labels)}")
    print(f"Sparse train shape: {X_train_sparse.shape}")
    print(f"Sparse test shape: {X_test_sparse.shape}")
    print(f"SVD components: {args.svd_components}")
    print(f"SVD train shape: {X_train_svd.shape}")
    print(f"SVD test shape: {X_test_svd.shape}")
    print(f"SMOTE rows before: {smote_rows_before}")
    print(f"SMOTE rows after: {smote_rows_after}")
    print(f"XGBoost mode used: {mode_used}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification report:")
    print(report)

    if args.save_artifacts:
        save_artifacts(
            joblib=joblib,
            output_dir=OUTPUT_DIR,
            vectorizer=vectorizer,
            svd=svd,
            label_encoder=label_encoder,
            model=model,
            report=report,
            metrics={
                "accuracy": accuracy,
                "xgboost_mode_used": mode_used,
                "total_fasta_records": dataset.total_count,
                "valid_genus_records": dataset.valid_count,
                "invalid_skipped_records": dataset.invalid_count,
                "top_genera": args.top_genera,
                "sample_size_used": len(labels),
                "sparse_train_shape": list(X_train_sparse.shape),
                "sparse_test_shape": list(X_test_sparse.shape),
                "svd_components": args.svd_components,
                "svd_train_shape": list(X_train_svd.shape),
                "svd_test_shape": list(X_test_svd.shape),
                "smote_rows_before": smote_rows_before,
                "smote_rows_after": smote_rows_after,
                "xgboost_params": build_xgboost_params(
                    args=args,
                    device="cuda" if mode_used == "gpu/cuda" else "cpu",
                    num_classes=len(label_encoder.classes_),
                ),
            },
        )
        print(f"Saved artifacts to: {OUTPUT_DIR}")

    return 0


def validate_args(args: argparse.Namespace) -> str | None:
    if args.top_genera < 1:
        return "--top-genera must be 1 or greater."
    if args.sample_size < 2:
        return "--sample-size must be 2 or greater."
    if args.svd_components < 1:
        return "--svd-components must be 1 or greater."
    if args.n_estimators < 1:
        return "--n-estimators must be 1 or greater."
    if args.max_depth < 1:
        return "--max-depth must be 1 or greater."
    if args.learning_rate <= 0:
        return "--learning-rate must be greater than 0."
    if not 0 < args.test_size < 1:
        return "--test-size must be between 0 and 1."
    return None


def train_xgboost(
    xgb,
    X_train,
    y_train,
    args: argparse.Namespace,
    device: str,
    num_classes: int,
):
    model = xgb.XGBClassifier(
        **build_xgboost_params(args=args, device=device, num_classes=num_classes)
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        model.fit(X_train, y_train)

    if device == "cuda":
        warning_text = "\n".join(str(item.message).lower() for item in caught_warnings)
        if any(term in warning_text for term in CUDA_FALLBACK_WARNING_TERMS):
            raise RuntimeError(warning_text)

    return model, "gpu/cuda" if device == "cuda" else "cpu"


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
        "tree_method": "hist",
        "device": device,
        "eval_metric": "mlogloss",
        "random_state": args.random_state,
        "n_jobs": -1,
    }


def save_artifacts(
    joblib,
    output_dir: Path,
    vectorizer,
    svd,
    label_encoder,
    model,
    report: str,
    metrics: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, output_dir / "vectorizer.joblib")
    joblib.dump(svd, output_dir / "svd.joblib")
    joblib.dump(label_encoder, output_dir / "label_encoder.joblib")
    joblib.dump(model, output_dir / "xgboost_model.joblib")
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
