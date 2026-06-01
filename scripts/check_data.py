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
        description="Verify that the configured FASTA file exists and is readable."
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
        "--records",
        type=int,
        default=5,
        help="Number of FASTA records to inspect. Defaults to 5.",
    )
    parser.add_argument(
        "--count-records",
        action="store_true",
        help="Count all FASTA records after the initial readability check.",
    )
    parser.add_argument(
        "--top-genera",
        type=int,
        metavar="N",
        help="Print the top N valid genera by record count.",
    )
    parser.add_argument(
        "--show-invalid",
        type=int,
        metavar="N",
        help="Show up to N invalid/skipped taxonomy examples.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from bacterial_classifier.config import load_config

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
        from bacterial_classifier.data import (
            count_genera,
            iter_fasta_records,
            load_fasta_dataset,
            select_top_genera,
        )
    except ImportError:
        print(
            "Biopython is not installed. Install dependencies with "
            "'pip install -r requirements.txt'.",
            file=sys.stderr,
        )
        return 2

    records_seen = 0
    try:
        for record in iter_fasta_records(fasta_path, limit=args.records):
            records_seen += 1
            print(
                f"{records_seen}. id={record['sequence_id']} "
                f"length={len(record['sequence'])} "
                f"taxonomy={record['taxonomy_label']} "
                f"genus={record['genus_label'] or '<invalid>'}"
            )
            if not record["is_valid"]:
                print(f"   skipped: {record['invalid_reason']}")
    except ImportError as exc:
        print(
            f"Missing dependency: {exc}. Install dependencies with "
            "'pip install -r requirements.txt'.",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        print(f"Biopython could not read the FASTA file: {exc}", file=sys.stderr)
        return 1

    if records_seen == 0:
        print("No FASTA records were found.", file=sys.stderr)
        return 1

    print(f"Read {records_seen} FASTA record(s) successfully.")

    if args.count_records or args.top_genera is not None or args.show_invalid is not None:
        try:
            dataset = load_fasta_dataset(fasta_path)
        except ImportError as exc:
            print(
                f"Missing dependency: {exc}. Install dependencies with "
                "'pip install -r requirements.txt'.",
                file=sys.stderr,
            )
            return 2
        except Exception as exc:
            print(f"Biopython could not read the FASTA file: {exc}", file=sys.stderr)
            return 1

        genus_counts = count_genera(dataset.genus_labels)

        if args.count_records:
            print(f"Total FASTA records: {dataset.total_count}")
            print(f"Valid genus records: {dataset.valid_count}")
            print(f"Invalid/skipped records: {dataset.invalid_count}")
            print(f"Unique valid genera: {len(genus_counts)}")

        if args.top_genera is not None:
            if args.top_genera < 1:
                print("--top-genera must be 1 or greater.", file=sys.stderr)
                return 2

            print(f"Top {args.top_genera} valid genera:")
            for rank, (genus, count) in enumerate(
                select_top_genera(dataset.genus_labels, args.top_genera),
                start=1,
            ):
                print(f"{rank}. {genus}: {count}")

        if args.show_invalid is not None:
            if args.show_invalid < 1:
                print("--show-invalid must be 1 or greater.", file=sys.stderr)
                return 2

            print(f"Invalid/skipped examples, up to {args.show_invalid}:")
            for rank, record in enumerate(
                dataset.invalid_records[: args.show_invalid],
                start=1,
            ):
                print(
                    f"{rank}. id={record.sequence_id} "
                    f"reason={record.reason} taxonomy={record.taxonomy_label}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
