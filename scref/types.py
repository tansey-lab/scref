"""Lightweight value types for the artifacts a reference produces.

`MarkerSet` and `WrongGeneSet` are thin wrappers around ``dict[str, set[str]]``
(cell type -> gene-symbol set). They exist so the API is self-documenting,
so artifacts carry a stable fingerprint, and so they pickle cleanly into a
caller-owned cache (scref itself never writes them to disk).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional, Set

import pandas as pd


def _fingerprint_gene_dict(d: Dict[str, Set[str]]) -> str:
    """Stable hash of a {cell_type: gene_set} mapping, invariant to set/dict
    iteration order. Two artifacts with identical contents always hash equal."""
    parts = []
    for ct in sorted(d):
        genes = ",".join(sorted(d[ct]))
        parts.append(f"{ct}:{genes}")
    blob = "|".join(parts).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


@dataclass
class MarkerSet:
    """Per-cell-type marker gene-symbol sets.

    `sources` records, per cell type, how its markers were obtained
    (e.g. ``"auto (37)"`` or ``"fallback (auto=2, lit=11)"``) for downstream
    diagnostics. It does not participate in the fingerprint.
    """

    genes: Dict[str, Set[str]]
    sources: Dict[str, str] = field(default_factory=dict)

    def __getitem__(self, cell_type: str) -> Set[str]:
        return self.genes[cell_type]

    def __contains__(self, cell_type: str) -> bool:
        return cell_type in self.genes

    def __iter__(self) -> Iterator[str]:
        return iter(self.genes)

    def get(self, cell_type: str, default: Optional[Set[str]] = None) -> Set[str]:
        if default is None:
            default = set()
        return self.genes.get(cell_type, default)

    @property
    def cell_types(self) -> list:
        return sorted(self.genes)

    @property
    def fingerprint(self) -> str:
        return _fingerprint_gene_dict(self.genes)

    def to_frame(self) -> pd.DataFrame:
        """Long-format (cell_type, gene) table, for inspection."""
        rows = [
            {"cell_type": ct, "gene": g}
            for ct in sorted(self.genes)
            for g in sorted(self.genes[ct])
        ]
        return pd.DataFrame(rows, columns=["cell_type", "gene"])


@dataclass
class WrongGeneSet:
    """Per-cell-type 'wrong gene' sets: markers of OTHER types that those other
    types express substantially more than this type does. A transcript of a
    wrong gene landing in a cell of this type is a contamination signal."""

    genes: Dict[str, Set[str]]

    def __getitem__(self, cell_type: str) -> Set[str]:
        return self.genes[cell_type]

    def __contains__(self, cell_type: str) -> bool:
        return cell_type in self.genes

    def __iter__(self) -> Iterator[str]:
        return iter(self.genes)

    def get(self, cell_type: str, default: Optional[Set[str]] = None) -> Set[str]:
        if default is None:
            default = set()
        return self.genes.get(cell_type, default)

    @property
    def cell_types(self) -> list:
        return sorted(self.genes)

    @property
    def fingerprint(self) -> str:
        return _fingerprint_gene_dict(self.genes)

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {"cell_type": ct, "gene": g}
            for ct in sorted(self.genes)
            for g in sorted(self.genes[ct])
        ]
        return pd.DataFrame(rows, columns=["cell_type", "gene"])
