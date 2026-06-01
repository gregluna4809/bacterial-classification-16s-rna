from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import CountVectorizer


DEFAULT_KMER_SIZES = (4, 5)


@dataclass(frozen=True)
class KmerAnalyzer:
    kmer_sizes: tuple[int, ...]

    def __call__(self, sequence: str) -> list[str]:
        return generate_kmers(sequence, self.kmer_sizes)


@dataclass(frozen=True)
class KmerFeatures:
    matrix: spmatrix
    labels: list[str]
    vectorizer: CountVectorizer

    @property
    def vocabulary_size(self) -> int:
        return len(self.vectorizer.vocabulary_)


def build_kmer_vectorizer(
    kmer_sizes: Sequence[int] = DEFAULT_KMER_SIZES,
    binary: bool = False,
) -> CountVectorizer:
    """Create a sparse k-mer count vectorizer for exactly the given k values."""
    normalized_sizes = normalize_kmer_sizes(kmer_sizes)
    return CountVectorizer(
        analyzer=KmerAnalyzer(normalized_sizes),
        lowercase=False,
        binary=binary,
    )


def generate_kmer_features(
    sequences: Sequence[str],
    labels: Sequence[str],
    kmer_sizes: Sequence[int] = DEFAULT_KMER_SIZES,
    binary: bool = False,
) -> KmerFeatures:
    """Generate sparse k-mer features without densifying the matrix."""
    if len(sequences) != len(labels):
        raise ValueError("sequences and labels must have the same length")

    vectorizer = build_kmer_vectorizer(kmer_sizes=kmer_sizes, binary=binary)
    matrix = vectorizer.fit_transform(sequences)
    return KmerFeatures(
        matrix=matrix,
        labels=list(labels),
        vectorizer=vectorizer,
    )


def generate_kmers(sequence: str, kmer_sizes: Iterable[int]) -> list[str]:
    """Return k-mers for the requested k values from one sequence."""
    normalized = sequence.upper()
    kmers: list[str] = []

    for kmer_size in kmer_sizes:
        if kmer_size > len(normalized):
            continue
        kmers.extend(
            normalized[start : start + kmer_size]
            for start in range(0, len(normalized) - kmer_size + 1)
        )

    return kmers


def normalize_kmer_sizes(kmer_sizes: Sequence[int]) -> tuple[int, ...]:
    """Validate and de-duplicate k-mer sizes while preserving sorted order."""
    if not kmer_sizes:
        raise ValueError("At least one k-mer size is required")

    normalized = tuple(sorted(set(int(size) for size in kmer_sizes)))
    invalid = [size for size in normalized if size < 1]
    if invalid:
        raise ValueError(f"k-mer sizes must be positive integers: {invalid}")

    return normalized
