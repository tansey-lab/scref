"""Shared kernels for the per-cell -> per-type metrics.

The headline aggregation everywhere is "mean of per-cell rates" (each cell
counts equally regardless of transcript load), NOT sum-over-sum. Centralizing
it here means that choice is made in exactly one place.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse


def _csr(adata: AnnData):
    X = adata.X
    return X.tocsr() if issparse(X) else X


def _gene_cols(adata: AnnData, genes: Sequence[str]) -> List[int]:
    present = [g for g in genes if g in adata.var_names]
    return [adata.var_names.get_loc(g) for g in present]


def _row_sums(X, cols: Optional[List[int]]) -> np.ndarray:
    """Per-row sum over a column subset (all columns if ``cols`` is None)."""
    if cols is not None and len(cols) == 0:
        return np.zeros(X.shape[0])
    Xs = X if cols is None else X[:, cols]
    if issparse(Xs):
        return np.asarray(Xs.sum(axis=1)).flatten()
    return np.asarray(Xs).sum(axis=1)


def per_cell_fraction_by_type(
    adata: AnnData,
    group_labels: np.ndarray,
    numerator_genes_by_type: Dict[str, Set[str]],
    *,
    value_name: str,
    min_cells: int = 50,
    per: str = "cell_type",
) -> pd.DataFrame:
    """Per cell of type T: ``sum(tx in numerator_genes[T]) / sum(all tx)``.

    ``per="cell_type"`` returns the per-type mean of those per-cell fractions
    (+ std, ddof=1) with diagnostic count totals. ``per="cell"`` returns one
    row per cell (only for types present in ``numerator_genes_by_type`` and
    meeting ``min_cells``). Cells with zero total tx are dropped.

    Used for both contamination_enrichment (numerator = wrong genes) and
    marker_enrichment (numerator = own-type markers).
    """
    if per not in ("cell", "cell_type"):
        raise ValueError(f"per must be 'cell'|'cell_type', got {per!r}")

    X = _csr(adata)
    groups = pd.Categorical(group_labels)
    per_cell_total = _row_sums(X, None)

    rows = []
    cell_rows = []
    for ct in groups.categories:
        if ct not in numerator_genes_by_type:
            continue
        mask = np.asarray(groups == ct)
        n = int(mask.sum())
        if n < min_cells:
            continue
        cell_idx = np.where(mask)[0]
        totals = per_cell_total[cell_idx]
        num_cols = _gene_cols(adata, numerator_genes_by_type[ct])
        per_cell_num = _row_sums(X[cell_idx, :], num_cols)

        valid = totals > 0
        if valid.sum() == 0:
            continue
        frac = per_cell_num[valid] / totals[valid]

        if per == "cell":
            vidx = cell_idx[valid]
            for k, ci in enumerate(vidx):
                cell_rows.append({
                    "cell_index": int(ci),
                    "cell_type": ct,
                    value_name: float(frac[k]),
                    "n_total_tx": float(totals[valid][k]),
                })
        else:
            rows.append({
                "cell_type": ct,
                "n_cells": n,
                f"n_{value_name}_tx": float(per_cell_num.sum()),
                "n_total_tx": float(totals.sum()),
                value_name: float(frac.mean()),
                f"{value_name}_std": (
                    float(frac.std(ddof=1)) if valid.sum() > 1 else 0.0),
            })

    if per == "cell":
        return pd.DataFrame(cell_rows,
                            columns=["cell_index", "cell_type",
                                     value_name, "n_total_tx"])
    return pd.DataFrame(
        rows,
        columns=["cell_type", "n_cells", f"n_{value_name}_tx",
                 "n_total_tx", value_name, f"{value_name}_std"])


def per_cell_ratio_by_type(
    adata: AnnData,
    group_labels: np.ndarray,
    numerator_genes_by_type: Dict[str, Set[str]],
    denominator_genes_by_type: Dict[str, Set[str]],
    *,
    min_cells: int = 50,
    per: str = "cell_type",
) -> pd.DataFrame:
    """Per cell of type T: ``sum(num[T] tx) / sum(denom[T] tx)``, restricted to
    cells where the denominator is > 0, then averaged per type.

    Used for contamination_to_marker_ratio (num = wrong, denom = own markers).
    """
    if per not in ("cell", "cell_type"):
        raise ValueError(f"per must be 'cell'|'cell_type', got {per!r}")

    X = _csr(adata)
    groups = pd.Categorical(group_labels)

    rows = []
    cell_rows = []
    for ct in groups.categories:
        if ct not in numerator_genes_by_type:
            continue
        mask = np.asarray(groups == ct)
        n = int(mask.sum())
        if n < min_cells:
            continue
        cell_idx = np.where(mask)[0]
        Xc = X[cell_idx, :]
        num = _row_sums(Xc, _gene_cols(adata, numerator_genes_by_type[ct]))
        den = _row_sums(Xc, _gene_cols(adata,
                                       denominator_genes_by_type.get(ct, set())))

        valid = den > 0
        n_valid = int(valid.sum())
        if n_valid < min_cells:
            continue
        ratio = num[valid] / den[valid]

        if per == "cell":
            vidx = cell_idx[valid]
            for k, ci in enumerate(vidx):
                cell_rows.append({
                    "cell_index": int(ci),
                    "cell_type": ct,
                    "ratio": float(ratio[k]),
                })
        else:
            rows.append({
                "cell_type": ct,
                "n_cells": n,
                "n_cells_with_denominator": n_valid,
                "n_numerator_tx": float(num.sum()),
                "n_denominator_tx": float(den.sum()),
                "ratio": float(ratio.mean()),
                "ratio_std": float(ratio.std(ddof=1)) if n_valid > 1 else 0.0,
            })

    if per == "cell":
        return pd.DataFrame(cell_rows,
                            columns=["cell_index", "cell_type", "ratio"])
    return pd.DataFrame(
        rows,
        columns=["cell_type", "n_cells", "n_cells_with_denominator",
                 "n_numerator_tx", "n_denominator_tx", "ratio", "ratio_std"])
