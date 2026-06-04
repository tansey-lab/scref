"""Per-cell-type marker construction from a scRNAseq reference.

Markers are built by ``rank_genes_groups`` (Wilcoxon) with a top-N + lfc-floor
selection rule, restricted to a panel-gene universe, with an optional
literature fallback for cell types that yield too few automatic markers.

The function takes counts already resolved into symbol space (see
``scref.preprocessing.counts_in_symbol_space``) rather than resolving them
itself, so the resolution happens exactly once per reference build.
"""

from __future__ import annotations

import sys
from typing import Dict, Optional, Set, Tuple

import scanpy as sc
from anndata import AnnData

from .types import MarkerSet

# Defaults validated against a 17-class harmonized vocabulary across many
# cancer datasets; every (dataset, cell_type) yielded a usable marker set.
DEFAULT_LFC_FLOOR = 2.0          # absolute log2 fold-change threshold (>= 4x)
DEFAULT_SCORE_QUANTILE = 0.50    # Wilcoxon-score quantile sanity gate
DEFAULT_MAX_MARKERS = 50         # top-N by score after lfc + padj filtering
DEFAULT_PVAL_CUTOFF = 0.01
DEFAULT_MIN_MARKERS_BEFORE_FALLBACK = 5
DEFAULT_MIN_CELLS_FOR_RANKING = 3
DEFAULT_MIN_REF_CELLS = 50       # types below this are excluded entirely


# A default literature panel for common immune/stromal/epithelial lineages.
# Injectable: pass your own ``literature_markers`` to ``build_markers`` to
# replace these for a different cell-type vocabulary.
DEFAULT_LITERATURE_MARKERS: Dict[str, list] = {
    "Neutrophil": ["S100A8", "S100A9", "ELANE", "MPO", "FCGR3B", "CSF3R",
                   "CXCR2", "FPR1", "CEACAM8", "LCN2", "LTF", "PRTN3",
                   "DEFA3", "DEFA4"],
    "NK": ["NKG7", "GNLY", "KLRD1", "KLRF1", "NCAM1", "FCGR3A", "PRF1",
           "GZMB", "GZMA"],
    "Mast": ["TPSAB1", "TPSB2", "CPA3", "KIT", "HDC", "MS4A2"],
    "DC": ["CD1C", "CLEC9A", "LAMP3", "FCER1A", "CLEC10A", "IRF8"],
    "Monocyte": ["CD14", "FCN1", "VCAN", "S100A12", "LYZ", "CSF1R"],
    "Macrophage": ["CD68", "CD163", "MRC1", "MSR1", "C1QA", "C1QB", "C1QC",
                   "APOE"],
    "B": ["MS4A1", "CD79A", "CD79B", "CD19", "CD22", "TNFRSF13C"],
    "Plasma": ["MZB1", "JCHAIN", "XBP1", "DERL3", "IGHG1", "IGHA1", "IGKC"],
    "CD4T": ["CD4", "IL7R", "FOXP3", "CTLA4", "TCF7", "CCR7"],
    "CD8T": ["CD8A", "CD8B", "GZMK", "GZMA", "GZMB", "PRF1", "NKG7"],
    "T_general": ["CD3D", "CD3E", "CD3G", "TRAC", "TRBC1", "TRBC2"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "ENG", "KDR"],
    "Fibroblast": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRA",
                   "FAP"],
    "SMC": ["ACTA2", "MYH11", "TAGLN", "CNN1", "DES", "MYLK"],
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "CLDN4"],
    "Tumor": ["EPCAM", "KRT8", "KRT18", "MKI67", "TOP2A"],
}


def build_markers(
    counts_symbol_adata: AnnData,
    *,
    annotation_col: str,
    lfc_floor: float = DEFAULT_LFC_FLOOR,
    score_quantile: float = DEFAULT_SCORE_QUANTILE,
    pval_cutoff: float = DEFAULT_PVAL_CUTOFF,
    max_markers: int = DEFAULT_MAX_MARKERS,
    min_markers_before_fallback: int = DEFAULT_MIN_MARKERS_BEFORE_FALLBACK,
    min_cells_for_ranking: int = DEFAULT_MIN_CELLS_FOR_RANKING,
    min_ref_cells: int = DEFAULT_MIN_REF_CELLS,
    panel_genes: Optional[Set[str]] = None,
    literature_markers: Optional[Dict[str, list]] = None,
    verbose: bool = True,
) -> MarkerSet:
    """Build per-cell-type marker sets.

    Parameters
    ----------
    counts_symbol_adata
        Raw counts indexed by gene symbol (use
        ``preprocessing.counts_in_symbol_space``). This function does NOT
        re-resolve counts; it normalizes and ranks the matrix it is given.
    annotation_col
        obs column holding the reference cell-type labels.
    panel_genes
        If given, counts are subset to these symbols BEFORE normalization and
        ranking, so all fold-changes / scores / quantiles are computed in the
        same gene universe the query will be scored in.
    literature_markers
        Fallback symbol lists per cell type, used when automatic ranking
        yields fewer than ``min_markers_before_fallback``. Defaults to
        ``DEFAULT_LITERATURE_MARKERS``; pass ``{}`` to disable fallback.

    Selection rule per cell type, in order:
      1. logfoldchange > ``lfc_floor`` AND padj < ``pval_cutoff``.
      2. score above the ``score_quantile`` percentile of the type's scores.
      3. top ``max_markers`` by score.

    Types with fewer than ``min_ref_cells`` cells are excluded entirely.
    """
    if literature_markers is None:
        literature_markers = DEFAULT_LITERATURE_MARKERS

    if annotation_col not in counts_symbol_adata.obs.columns:
        from .exceptions import AnnotationColumnError
        raise AnnotationColumnError(
            f"annotation_col {annotation_col!r} not in obs "
            f"({list(counts_symbol_adata.obs.columns)})"
        )

    all_types = sorted(counts_symbol_adata.obs[annotation_col].unique())
    type_counts = counts_symbol_adata.obs[annotation_col].value_counts()

    included_types = [t for t in all_types
                      if type_counts.get(t, 0) >= min_ref_cells]
    excluded_types = [t for t in all_types if t not in included_types]
    if excluded_types and verbose:
        summary = ", ".join(
            f"{t}({type_counts.get(t, 0)})" for t in excluded_types)
        print(f"[scref.markers] excluded (min_ref_cells={min_ref_cells}): "
              f"{summary}", file=sys.stderr)

    rankable_types = [t for t in included_types
                      if type_counts.get(t, 0) >= min_cells_for_ranking]
    dropped_from_ranking = [t for t in included_types
                            if t not in rankable_types]

    # Subset to rankable cells, then panel genes, BEFORE normalize/rank.
    counts = counts_symbol_adata[
        counts_symbol_adata.obs[annotation_col].isin(rankable_types)
    ].copy()
    if panel_genes is not None:
        panel_in_var = [g for g in counts.var_names if g in panel_genes]
        counts = counts[:, panel_in_var].copy()
    sc.pp.normalize_total(counts, target_sum=1e4)
    sc.pp.log1p(counts)

    rank_worked = False
    if len(rankable_types) >= 2:
        try:
            sc.tl.rank_genes_groups(counts, groupby=annotation_col,
                                    method="wilcoxon", use_raw=False)
            rank_worked = True
        except Exception:
            rank_worked = False

    # Reference symbol universe, for intersecting literature fallbacks.
    ref_symbols = set(counts_symbol_adata.var_names)

    markers: Dict[str, Set[str]] = {}
    sources: Dict[str, str] = {}
    for ct in included_types:
        auto: Set[str] = set()
        if rank_worked and ct in rankable_types:
            try:
                result = sc.get.rank_genes_groups_df(counts, group=ct)
                cand = result[
                    (result["logfoldchanges"] > lfc_floor)
                    & (result["pvals_adj"] < pval_cutoff)
                ]
                score_cutoff = result["scores"].quantile(score_quantile)
                cand = cand[cand["scores"] > score_cutoff]
                cand = cand.head(max_markers)
                auto = set(cand["names"].tolist())
            except Exception:
                auto = set()

        if len(auto) < min_markers_before_fallback and ct in literature_markers:
            lit = set(literature_markers[ct]) & ref_symbols
            if panel_genes is not None:
                lit = lit & panel_genes
            markers[ct] = auto | lit
            if ct in dropped_from_ranking:
                sources[ct] = (
                    f"literature-only (had {type_counts.get(ct, 0)} cells, "
                    f"< min_cells_for_ranking={min_cells_for_ranking}; "
                    f"lit={len(lit)})")
            else:
                sources[ct] = f"fallback (auto={len(auto)}, lit={len(lit)})"
        else:
            markers[ct] = auto
            sources[ct] = f"auto ({len(auto)})"

    return MarkerSet(genes=markers, sources=sources)
