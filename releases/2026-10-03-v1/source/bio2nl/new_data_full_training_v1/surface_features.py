"""Pure, symmetric B6 features over existing shared-BPE content token IDs.

This module performs no I/O, tokenization, text normalization, model inference,
or statistical fitting. Every endpoint descriptor depends only on its own IDs.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np


FEATURE_COLUMNS = [
    "length_log1p_min", "length_log1p_max", "length_min_max_ratio",
    "unigram_set_jaccard", "unigram_multiset_dice", "unigram_count_cosine",
    "bigram_set_jaccard", "bigram_multiset_dice", "bigram_count_cosine",
]
FAMILY_COLUMNS = {
    "length": FEATURE_COLUMNS[:3],
    "unigram": FEATURE_COLUMNS[3:6],
    "length_unigram": FEATURE_COLUMNS[:6],
    "length_unigram_bigram": FEATURE_COLUMNS.copy(),
}


@dataclass(frozen=True, slots=True)
class _Multiset:
    counts: Mapping
    unique: frozenset
    total: int
    squared_norm: int


@dataclass(frozen=True, slots=True)
class Endpoint:
    """Reusable, read-only counts for one validated content-token sequence."""
    length: int
    unigram: _Multiset
    bigram: _Multiset


def _multiset(items):
    counts = Counter(items)
    return _Multiset(MappingProxyType(dict(counts)), frozenset(counts),
                     sum(counts.values()), sum(count * count for count in counts.values()))


def endpoint(sequence):
    """Validate exact integer content IDs and compute each endpoint's counts once."""
    try:
        ids = tuple(sequence)
    except TypeError as error:
        raise ValueError("A nonempty sequence of exact integer content IDs is required") from error
    if not ids or any(type(token) is not int or not 4 <= token < 32000 for token in ids):
        raise ValueError("Content IDs must be nonempty exact integers in [4, 32000)")
    return Endpoint(len(ids), _multiset(ids), _multiset(zip(ids, ids[1:])))


def _overlap(first, second):
    if not first.total and not second.total:
        return 1., 1., 1.
    if not first.total or not second.total:
        return 0., 0., 0.
    shared = first.unique & second.unique
    jaccard = len(shared) / (len(first.unique) + len(second.unique) - len(shared))
    intersection = sum(min(first.counts[key], second.counts[key]) for key in shared)
    dice = 2 * intersection / (first.total + second.total)
    dot = sum(first.counts[key] * second.counts[key] for key in shared)
    cosine = dot / math.sqrt(first.squared_norm * second.squared_norm)
    return jaccard, dice, cosine


def _vector(first, second):
    small, large = min(first.length, second.length), max(first.length, second.length)
    result = np.asarray((math.log1p(small), math.log1p(large), small / large,
                         *_overlap(first.unigram, second.unigram),
                         *_overlap(first.bigram, second.bigram)), dtype=np.float64)
    if result.shape != (9,) or not np.isfinite(result).all():
        raise ValueError("Surface features must be finite float64 values with nine columns")
    return result


def vector_features(ids_a, ids_b):
    """Nine features for one unordered endpoint pair, in FEATURE_COLUMNS order."""
    return _vector(endpoint(ids_a), endpoint(ids_b))


def features_for_pairs(content_ids, pairs):
    """Compute FP64[N,9] from content-ID sequences and pair dictionaries with a/b.

    Endpoint counters are computed once and reused across all pairs. Extra pair
    metadata, including labels, is ignored; provenance and split validation are
    the responsibility of the caller. Empty pair batches return shape (0, 9).
    """
    descriptors = [endpoint(ids) for ids in content_ids]
    rows = list(pairs)
    result = np.empty((len(rows), len(FEATURE_COLUMNS)), dtype=np.float64)
    for index, row in enumerate(rows):
        try:
            a, b = row["a"], row["b"]
        except (KeyError, TypeError) as error:
            raise ValueError("Pair dictionaries must supply endpoint indices a and b") from error
        if (type(a) is not int or type(b) is not int or
                min(a, b) < 0 or max(a, b) >= len(descriptors)):
            raise ValueError("Endpoint indices must be exact integers inside the sequence list")
        result[index] = _vector(descriptors[a], descriptors[b])
    return result
