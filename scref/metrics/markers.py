"""Marker metrics: marker enrichment (per-cell) and marker specificity
(per-gene tx accounting)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse

from ..types import MarkerSet
from ._shared import per_cell_fraction_by_type


def marker_enrichment(
    adata: AnnData,
    group_labels: np.ndarray,
    markers: MarkerSet,
    *,
    min_cells: int = 50,
    per: str = "cell_type",
) -> pd.DataFrame:
    """Fraction of a cell's transcripts that fall in its OWN type's markers.
    Symmetric to contamination_enrichment (own markers vs wrong genes).

    ``per="cell_type"`` columns:
        cell_type, n_cells, n_enrichment_tx, n_total_tx,
        enrichment, enrichment_std
    ``per="cell"`` columns:
        cell_index, cell_type, enrichment, n_total_tx
    """
    return per_cell_fraction_by_type(
        adata, group_labels, markers.genes,
        value_name="enrichment", min_cells=min_cells, per=per)


def marker_specificity(
    adata: AnnData,
    group_labels: np.ndarray,
    markers: MarkerSet,
    *,
    min_total_counts: int = 50,
) -> pd.DataFrame:
    """For each (cell type T, T-marker g): fraction of g's total transcripts
    (across all cells) that land in cells typed as T.

    This is a gene-accounting metric, so it has no per-cell form. Markers
    whose genome-wide total is below ``min_total_counts`` are skipped.

    Columns: cell_type, marker, total_counts, correct_counts, specificity
    """
    X = adata.X
    if issparse(X):
        X = X.tocsr()
    total_per_gene = (np.asarray(X.sum(axis=0)).flatten()
                      if issparse(X) else np.asarray(X).sum(axis=0))
    gene_to_col = {g: i for i, g in enumerate(adata.var_names)}
    groups = pd.Categorical(group_labels)

    per_type_sums = {}
    for ct in groups.categories:
        mask = np.asarray(groups == ct)
        if int(mask.sum()) == 0:
            continue
        Xc = X[np.where(mask)[0], :]
        per_type_sums[ct] = (np.asarray(Xc.sum(axis=0)).flatten()
                             if issparse(Xc) else np.asarray(Xc).sum(axis=0))

    rows = []
    for ct in markers:
        if ct not in per_type_sums:
            continue
        for g in markers[ct]:
            if g not in gene_to_col:
                continue
            j = gene_to_col[g]
            total = float(total_per_gene[j])
            if total < min_total_counts:
                continue
            correct = float(per_type_sums[ct][j])
            rows.append({
                "cell_type": ct,
                "marker": g,
                "total_counts": total,
                "correct_counts": correct,
                "specificity": correct / total if total > 0 else np.nan,
            })
    return pd.DataFrame(
        rows,
        columns=["cell_type", "marker", "total_counts",
                 "correct_counts", "specificity"])
