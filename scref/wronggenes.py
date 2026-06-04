"""Wrong-gene construction: for each cell type T, the markers of OTHER types
that those other types express substantially more than T does.

A transcript of a wrong gene found inside a T-typed cell is a contamination
signal. The selection is symmetric with marker construction: markers of T are
genes T expresses >= 2^lfc more than others; wrong-genes for T are markers of
others that others express >= 2^wrong_lfc more than T.

Means are computed on log1p-normalized counts in symbol space, so the lfc
threshold (a difference of means in natural-log space) averages over dropout
and is robust to reference size, unlike a fraction-expressing filter.
"""

from __future__ import annotations

from typing import Dict, Optional, Set

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import issparse

from .types import MarkerSet, WrongGeneSet

DEFAULT_WRONG_LFC = 2.0           # log2 negative-lfc threshold (4x)
DEFAULT_FRAC_IN_TARGET_FLOOR = 0.25


def build_wrong_genes(
    counts_symbol_adata: AnnData,
    markers: MarkerSet,
    *,
    annotation_col: str,
    wrong_lfc_threshold: float = DEFAULT_WRONG_LFC,
    frac_expr_in_target_floor: float = DEFAULT_FRAC_IN_TARGET_FLOOR,
) -> WrongGeneSet:
    """Build per-cell-type wrong-gene sets.

    Selection rule for ``wrong[T]``::

        g in wrong[T] iff
            g in union(markers[T'] for T' != T)
            g not in markers[T]
            max_{T': g in markers[T']} mean_expr(g, T') - mean_expr(g, T)
                > wrong_lfc_threshold * ln(2)
            AND  frac of T's cells expressing g  <  frac_expr_in_target_floor

    The strongest source type is used for the comparison (most conservative:
    hardest to flag as wrong). The fraction-in-target floor removes genes T
    genuinely and frequently expresses (not a credible contamination signal);
    set to 0.0 to disable.

    Parameters
    ----------
    counts_symbol_adata
        Raw counts in symbol space (NOT pre-normalized). Normalized once here.
    markers
        The marker sets the wrong genes are derived from.
    """
    marker_genes = markers.genes
    all_types = list(marker_genes.keys())
    if not any(marker_genes[t] for t in all_types):
        return WrongGeneSet(genes={t: set() for t in all_types})

    adata_norm = counts_symbol_adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)

    X = adata_norm.X
    if issparse(X):
        X = X.tocsr()
    groups = pd.Categorical(adata_norm.obs[annotation_col].values)
    gene_names = list(adata_norm.var_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    mean_by_type: Dict[str, np.ndarray] = {}
    frac_by_type: Dict[str, np.ndarray] = {}
    for ct in groups.categories:
        mask = groups == ct
        if int(mask.sum()) == 0:
            continue
        Xc = X[np.where(mask)[0], :]
        if issparse(Xc):
            mean_by_type[ct] = np.asarray(Xc.mean(axis=0)).flatten()
            frac_by_type[ct] = np.asarray((Xc > 0).mean(axis=0)).flatten()
        else:
            mean_by_type[ct] = Xc.mean(axis=0)
            frac_by_type[ct] = (Xc > 0).mean(axis=0)

    # Per gene, which types it markers (to find each candidate's source(s)).
    gene_sources: Dict[str, list] = {}
    for t, mk in marker_genes.items():
        for g in mk:
            gene_sources.setdefault(g, []).append(t)

    nat_log_threshold = wrong_lfc_threshold * np.log(2)

    wrong: Dict[str, Set[str]] = {}
    for ct in all_types:
        own_markers = marker_genes.get(ct, set())
        union_other: Set[str] = set()
        for other in all_types:
            if other != ct:
                union_other |= marker_genes[other]
        union_other -= own_markers

        if ct not in mean_by_type:
            # Can't characterize this type's expression; fall back to the
            # unfiltered union so the type isn't silently dropped.
            wrong[ct] = union_other
            continue

        m_t = mean_by_type[ct]
        f_t = frac_by_type.get(ct)
        kept: Set[str] = set()
        for g in union_other:
            if g not in gene_to_idx:
                continue
            idx = gene_to_idx[g]
            if (frac_expr_in_target_floor > 0.0 and f_t is not None
                    and f_t[idx] >= frac_expr_in_target_floor):
                continue
            t_expr = m_t[idx]
            sources = [s for s in gene_sources.get(g, []) if s != ct]
            present = [s for s in sources if s in mean_by_type]
            if not present:
                continue
            max_source_expr = max(mean_by_type[s][idx] for s in present)
            if (max_source_expr - t_expr) > nat_log_threshold:
                kept.add(g)
        wrong[ct] = kept

    return WrongGeneSet(genes=wrong)
