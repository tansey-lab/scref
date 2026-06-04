"""DE concordance: per cell-type pair, the Spearman correlation between the
reference's per-gene log-fold-change (A vs B) and the query's.

The reference side is captured once at reference-build time as a `Pseudobulk`
(per-type mean on log1p-normalized expression + per-type fraction-expressing),
so a reference reloaded from a cache can compute concordance against a new
query without the original reference AnnData.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import stats
from scipy.sparse import issparse

from ..preprocessing import normalize_log1p


@dataclass
class Pseudobulk:
    """Per-cell-type pseudobulk over a fixed gene vector.

    ``mean`` / ``frac_expr`` map cell type -> length-``len(genes)`` arrays on
    the SAME gene order (``genes``), so two pseudobulks built on a shared gene
    space are directly comparable column-for-column.
    """

    genes: np.ndarray
    mean: Dict[str, np.ndarray]
    frac_expr: Dict[str, np.ndarray]
    n: Dict[str, int]


def build_pseudobulk(
    adata: AnnData,
    *,
    group_col: str,
    genes: np.ndarray,
    min_cells: int,
    already_normalized: bool = False,
) -> Pseudobulk:
    """Build a `Pseudobulk` over ``genes`` (in that order). Types with fewer
    than ``min_cells`` cells are omitted.

    ``already_normalized=False`` (default) normalizes + log1p's internally;
    pass True if ``adata.X`` is already log1p-normalized.
    """
    genes = np.asarray([g for g in genes if g in set(adata.var_names)])
    sub = adata[:, genes].copy()
    norm = sub if already_normalized else normalize_log1p(sub)

    X = norm.X
    if issparse(X):
        X = X.tocsr()
    groups = pd.Categorical(norm.obs[group_col].values)

    mean: Dict[str, np.ndarray] = {}
    frac: Dict[str, np.ndarray] = {}
    nmap: Dict[str, int] = {}
    for ct in groups.categories:
        mask = np.asarray(groups == ct)
        n = int(mask.sum())
        if n < min_cells:
            continue
        Xc = X[np.where(mask)[0], :]
        if issparse(Xc):
            mean[ct] = np.asarray(Xc.mean(axis=0)).flatten()
            frac[ct] = np.asarray((Xc > 0).mean(axis=0)).flatten()
        else:
            mean[ct] = np.asarray(Xc).mean(axis=0)
            frac[ct] = (np.asarray(Xc) > 0).mean(axis=0)
        nmap[ct] = n
    return Pseudobulk(genes=genes, mean=mean, frac_expr=frac, n=nmap)


def de_concordance(
    reference_pseudobulk: Pseudobulk,
    query: AnnData,
    *,
    query_group_col: str,
    min_cells: int = 50,
    min_expr_frac: float = 0.05,
) -> pd.DataFrame:
    """Spearman DE concordance per cell-type pair.

    For each pair (A, B) present in both the reference pseudobulk and the
    query (after ``min_cells`` filtering), restrict to genes expressed in
    >= ``min_expr_frac`` of A or B in the REFERENCE, then correlate the
    reference LFC (mean_A - mean_B) with the query LFC over those genes.

    The query is pseudobulked on the reference's exact gene vector/order so
    the two LFC vectors align column-for-column.

    Columns: type_a, type_b, pair, n_cells_a, n_cells_b, n_genes, spearman
    """
    ref = reference_pseudobulk
    query_pb = build_pseudobulk(
        query, group_col=query_group_col, genes=ref.genes, min_cells=min_cells)

    # The reference pseudobulk was built on the reference's gene universe, but
    # the query may carry only a subset of those genes (build_pseudobulk drops
    # requested genes the query lacks, so query_pb.genes can be shorter). The
    # two mean/frac vectors therefore live on DIFFERENT axes and cannot be
    # indexed with a shared mask. Align both to the genes present in BOTH, by
    # NAME, before masking — this reproduces contamination_analyses, which
    # pre-intersects the reference and spatial adatas to their common genes so
    # only shared genes ever enter the correlation.
    qry_gene_set = set(query_pb.genes)
    shared_genes = np.array([g for g in ref.genes if g in qry_gene_set])
    ref_pos = {g: i for i, g in enumerate(ref.genes)}
    qry_pos = {g: i for i, g in enumerate(query_pb.genes)}
    ref_idx = np.array([ref_pos[g] for g in shared_genes])
    qry_idx = np.array([qry_pos[g] for g in shared_genes])

    shared = sorted(set(ref.mean) & set(query_pb.mean))
    rows: List[dict] = []
    for i, a in enumerate(shared):
        for b in shared[i + 1:]:
            # Reference frac_expr restricted to shared genes, then the
            # min_expr_frac filter (computed on the reference, as in ca).
            expr_a = ref.frac_expr[a][ref_idx]
            expr_b = ref.frac_expr[b][ref_idx]
            use = (expr_a >= min_expr_frac) | (expr_b >= min_expr_frac)
            if use.sum() < 10:
                continue
            ref_lfc = (ref.mean[a][ref_idx][use] - ref.mean[b][ref_idx][use])
            qry_lfc = (query_pb.mean[a][qry_idx][use]
                       - query_pb.mean[b][qry_idx][use])
            rho = stats.spearmanr(ref_lfc, qry_lfc).correlation
            rows.append({
                "type_a": a,
                "type_b": b,
                "pair": f"{a} vs {b}",
                "n_cells_a": query_pb.n[a],
                "n_cells_b": query_pb.n[b],
                "n_genes": int(use.sum()),
                "spearman": rho,
            })
    return pd.DataFrame(
        rows,
        columns=["type_a", "type_b", "pair", "n_cells_a", "n_cells_b",
                 "n_genes", "spearman"])