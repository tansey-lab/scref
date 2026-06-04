"""Input-state normalization for AnnData objects.

The single most bug-prone contract in this pipeline is "where do the counts
live and what id space are the var_names in". `counts_in_symbol_space` is the
one place that resolves it, so every downstream computation (marker ranking,
wrong-gene means, metric numerators) operates on raw counts indexed by gene
symbol regardless of how the input was preprocessed.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse

from .exceptions import GeneSpaceError


def counts_in_symbol_space(
    adata: AnnData,
    *,
    raw: str = "auto",
    feature_name_col: str = "feature_name",
) -> AnnData:
    """Return a new AnnData whose ``.X`` holds raw COUNTS and whose var_names
    are gene SYMBOLS.

    Resolution contract:
      - Counts live in ``.raw`` if present (and ``raw`` allows it), else ``.X``.
      - For whichever var object accompanies the counts, if it has a
        ``feature_name_col`` column that column holds symbols; otherwise the
        var index already holds symbols.
      - Duplicate symbols (several ids mapping to one symbol) are dropped,
        keeping first occurrence.

    Parameters
    ----------
    raw : {"auto", "always", "never"}
        ``"auto"`` uses ``.raw`` when it exists (the common scRNAseq layout
        with a transformed ``.X`` and a counts ``.raw``). ``"never"`` forces
        use of ``.X``; ``"always"`` requires ``.raw`` and errors if absent.

    The returned object carries no ``.raw`` so that downstream scanpy calls
    with ``use_raw=False`` have nothing to fall back to.
    """
    if raw not in ("auto", "always", "never"):
        raise ValueError(f"raw must be 'auto'|'always'|'never', got {raw!r}")

    use_raw = (raw == "always") or (raw == "auto" and adata.raw is not None)
    if raw == "always" and adata.raw is None:
        raise GeneSpaceError("raw='always' but adata has no .raw layer")

    if use_raw:
        X = adata.raw.X
        var = adata.raw.var
    else:
        X = adata.X
        var = adata.var

    if feature_name_col in var.columns:
        symbols = pd.Index(var[feature_name_col].astype(str))
    else:
        symbols = pd.Index(var.index.astype(str))

    keep = ~symbols.duplicated(keep="first")
    Xk = X[:, np.asarray(keep)]
    out = AnnData(
        X=(Xk.tocsr().copy() if hasattr(Xk, "tocsr") else np.asarray(Xk).copy()),
        obs=adata.obs.copy(),
    )
    out.var_names = symbols[keep]
    out.raw = None
    return out


def normalize_log1p(adata: AnnData, target_sum: float = 1e4) -> AnnData:
    """Return a normalized + log1p'd copy. Operates on a copy; input untouched."""
    out = adata.copy()
    sc.pp.normalize_total(out, target_sum=target_sum)
    sc.pp.log1p(out)
    return out


def intersect_genes(
    reference_genes,
    query: AnnData,
    *,
    min_shared: int = 1,
) -> np.ndarray:
    """Genes shared between a reference gene universe and a query AnnData,
    in sorted order. Raises if fewer than ``min_shared`` are common."""
    ref = set(map(str, reference_genes))
    qry = set(map(str, query.var_names))
    common = np.array(sorted(ref & qry))
    if len(common) < min_shared:
        raise GeneSpaceError(
            f"only {len(common)} shared genes between reference and query "
            f"(min_shared={min_shared}); check the var_names id space"
        )
    return common


def reindex_to_panel(
    adata: AnnData,
    panel: Sequence[str],
    *,
    fill: float = 0.0,
) -> AnnData:
    """Return a new AnnData whose var axis is exactly ``panel`` (that order),
    zero-filling (or ``fill``-filling) any panel gene the input lacks.

    This is for scoring several differently-paneled query objects on ONE fixed
    gene panel. The motivating case: a segmentation method (e.g. Segger) emits
    a cell x gene matrix that omits genes for which it assigned zero transcripts
    to any cell. Those genes are genuinely all-zero columns in that method's
    matrix, so restoring them as zero columns reconstructs the true value and
    puts every method on the same panel — making per-cell denominators and
    pseudobulk normalization totals comparable across methods.

    IMPORTANT: only use this when an absent gene legitimately means "zero counts
    in this object". If a gene is absent because the object simply did not
    measure it (a different panel, not a dropped empty column), zero-filling
    fabricates data — verify the absence semantics first.

    Genes in ``adata`` that are NOT in ``panel`` are dropped. obs is preserved.
    """
    panel = list(map(str, panel))
    panel_set = set(panel)
    if len(panel_set) != len(panel):
        raise GeneSpaceError("panel contains duplicate gene symbols")

    var_names = list(map(str, adata.var_names))
    pos = {g: i for i, g in enumerate(var_names)}

    n_obs = adata.n_obs
    X = adata.X
    is_sparse = sparse.issparse(X)

    # Columns of the input to copy, and where each lands in the output.
    present_cols = [pos[g] for g in panel if g in pos]
    present_out = [j for j, g in enumerate(panel) if g in pos]
    n_present = len(present_cols)
    n_missing = len(panel) - n_present

    if is_sparse:
        Xin = X.tocsc()
        if fill != 0.0:
            # A nonzero fill makes the matrix dense by definition; honor it but
            # don't pretend it stays sparse.
            out = np.full((n_obs, len(panel)), float(fill))
            if n_present:
                sub = Xin[:, present_cols].toarray()
                out[:, present_out] = sub
            out_X = out
        else:
            out_X = sparse.lil_matrix((n_obs, len(panel)), dtype=Xin.dtype)
            if n_present:
                out_X[:, present_out] = Xin[:, present_cols]
            out_X = out_X.tocsr()
    else:
        Xin = np.asarray(X)
        out_X = np.full((n_obs, len(panel)), float(fill), dtype=Xin.dtype)
        if n_present:
            out_X[:, present_out] = Xin[:, present_cols]

    out = AnnData(X=out_X, obs=adata.obs.copy())
    out.var_names = pd.Index(panel)
    return out