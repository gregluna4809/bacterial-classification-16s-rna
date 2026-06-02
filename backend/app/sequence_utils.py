from __future__ import annotations

import re


VALID_SEQUENCE_PATTERN = re.compile(r"^[ACGTN]+$")


def clean_sequence(sequence: str) -> str:
    return "".join(str(sequence).split()).upper()


def validate_sequence(sequence: str, min_length: int = 100) -> str:
    cleaned = clean_sequence(sequence)
    if not cleaned:
        raise ValueError("Sequence is empty after removing whitespace.")
    if len(cleaned) < min_length:
        raise ValueError(f"Sequence must be at least {min_length} bases long.")
    if not VALID_SEQUENCE_PATTERN.fullmatch(cleaned):
        raise ValueError("Sequence may only contain A, C, G, T, and N.")
    return cleaned
