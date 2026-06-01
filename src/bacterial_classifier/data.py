from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Iterable


PLACEHOLDER_GENUS_LABELS = {
    "",
    "root",
    "bacteria",
    "archaea",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "unknown",
    "unclassified",
    "uncultured",
    "candidatus",
    "norank",
    "no_rank",
}
GROUP_PLACEHOLDER_PREFIXES = ("gp",)
NON_GENUS_SUFFIXES = ("aceae", "ales", "ineae", "phyta")
KNOWN_HIGHER_RANK_LABELS = {
    "acidobacteria",
    "actinobacteria",
    "alphaproteobacteria",
    "bacilli",
    "bacteria",
    "bacteroidia",
    "betaproteobacteria",
    "clostridia",
    "cyanobacteria",
    "deltaproteobacteria",
    "epsilonproteobacteria",
    "firmicutes",
    "gammaproteobacteria",
    "proteobacteria",
}


@dataclass(frozen=True)
class InvalidFastaRecord:
    sequence_id: str
    taxonomy_label: str
    description: str
    reason: str


@dataclass(frozen=True)
class FastaDataset:
    sequence_ids: list[str]
    sequences: list[str]
    taxonomy_labels: list[str]
    genus_labels: list[str]
    invalid_records: list[InvalidFastaRecord]

    @property
    def valid_count(self) -> int:
        return len(self.sequence_ids)

    @property
    def invalid_count(self) -> int:
        return len(self.invalid_records)

    @property
    def total_count(self) -> int:
        return self.valid_count + self.invalid_count


def load_fasta_dataset(path: Path | str, limit: int | None = None) -> FastaDataset:
    """Parse valid FASTA records and retain invalid taxonomy rows separately."""
    sequence_ids: list[str] = []
    sequences: list[str] = []
    taxonomy_labels: list[str] = []
    genus_labels: list[str] = []
    invalid_records: list[InvalidFastaRecord] = []

    for record in iter_fasta_records(path, limit=limit):
        if not record["is_valid"]:
            invalid_records.append(
                InvalidFastaRecord(
                    sequence_id=record["sequence_id"],
                    taxonomy_label=record["taxonomy_label"],
                    description=record["description"],
                    reason=record["invalid_reason"],
                )
            )
            continue

        sequence_ids.append(record["sequence_id"])
        sequences.append(record["sequence"])
        taxonomy_labels.append(record["taxonomy_label"])
        genus_labels.append(record["genus_label"])

    return FastaDataset(
        sequence_ids=sequence_ids,
        sequences=sequences,
        taxonomy_labels=taxonomy_labels,
        genus_labels=genus_labels,
        invalid_records=invalid_records,
    )


def iter_fasta_records(path: Path | str, limit: int | None = None):
    """Yield parsed FASTA records as dictionaries."""
    from Bio import SeqIO

    record_iter = SeqIO.parse(str(path), "fasta")
    if limit is not None:
        record_iter = islice(record_iter, limit)

    for record in record_iter:
        taxonomy_label = parse_taxonomy_label(record.description)
        genus_label, invalid_reason = parse_genus_label(taxonomy_label)
        yield {
            "sequence_id": record.id,
            "sequence": str(record.seq),
            "taxonomy_label": taxonomy_label,
            "genus_label": genus_label,
            "description": record.description,
            "is_valid": invalid_reason is None,
            "invalid_reason": invalid_reason,
        }


def parse_taxonomy_label(description: str) -> str:
    """Extract the taxonomy label from an RDP-style FASTA description."""
    parts = description.split()
    if len(parts) >= 2:
        return parts[1]
    if parts:
        return parts[0]
    return "unknown"


def parse_genus_label(taxonomy_label: str) -> tuple[str | None, str | None]:
    """Return the rightmost valid genus-like taxonomy token, or an invalid reason."""
    normalized = taxonomy_label.replace("|", ";")
    parts = [part.strip() for part in normalized.split(";")]

    rejected: list[str] = []
    for candidate in reversed(parts):
        genus = _clean_taxonomy_token(candidate)
        invalid_reason = _invalid_genus_reason(genus)
        if invalid_reason is None:
            return genus, None
        rejected.append(f"{genus or '<empty>'} ({invalid_reason})")

    if not parts or all(not part.strip() for part in parts):
        return None, "missing taxonomy tokens"

    return None, "no valid genus-like token found"


def _clean_taxonomy_token(token: str) -> str:
    cleaned = token.strip()
    for prefix in ("genus:", "g__", "Genus_"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    return cleaned.strip()


def _invalid_genus_reason(genus: str) -> str | None:
    normalized = genus.strip().lower()
    if normalized in PLACEHOLDER_GENUS_LABELS:
        return "placeholder genus label"
    if normalized.startswith("candidatus"):
        return "Candidatus placeholder/provisional label"
    if "_gp" in normalized:
        return "rank/group placeholder"
    if _contains_excluded_label_text(normalized):
        return "unclean environmental or uncertain label"
    if "/" in genus:
        return "slash-delimited non-genus label"
    if "chloroplast" in normalized:
        return "chloroplast-derived label"
    if "mitochondria" in normalized:
        return "mitochondria-derived label"
    if _is_group_placeholder(normalized):
        return "rank/group placeholder"
    if normalized in KNOWN_HIGHER_RANK_LABELS:
        return "known higher-rank label"
    if normalized.endswith(NON_GENUS_SUFFIXES):
        return "higher-rank suffix"
    if normalized.endswith("ia") and _is_class_like_ia_label(normalized):
        return "class-like suffix"
    return None


def _is_group_placeholder(normalized: str) -> bool:
    for prefix in GROUP_PLACEHOLDER_PREFIXES:
        suffix = normalized.removeprefix(prefix)
        if suffix and suffix.isdigit():
            return True
        suffix_index = normalized.rfind(prefix)
        if suffix_index >= 0 and normalized[suffix_index + len(prefix) :].isdigit():
            return True
    return False


def _is_class_like_ia_label(normalized: str) -> bool:
    return normalized.endswith("bacteria")


def _contains_excluded_label_text(normalized: str) -> bool:
    excluded_terms = (
        "incertae",
        "incertae_sedis",
        "genera_incertae_sedis",
        "unidentified",
        "metagenome",
        "environmental",
    )
    return any(term in normalized for term in excluded_terms)


def count_genera(genus_labels: Iterable[str]) -> Counter[str]:
    """Count records per genus label."""
    return Counter(genus_labels)


def select_top_genera(genus_labels: Iterable[str], top_n: int) -> list[tuple[str, int]]:
    """Return the top N genera as (genus, count) pairs."""
    return count_genera(genus_labels).most_common(top_n)
