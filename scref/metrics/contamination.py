"""Contamination metrics: wrong-gene enrichment and wrong-to-marker ratio."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from ..types import MarkerSet, WrongGeneSet
from ._shared import per_cell_fraction_by_type, per_cell_ratio_by_type


def contamination_enrichment(
    adata: AnnData,
    group_labels: np.ndarray,
    wrong_genes: WrongGeneSet,
    *,
    min_cells: int = 50,
    per: str = "cell_type",
) -> pd.DataFrame:
    """Fraction of a cell's transcripts that are wrong genes for its type.

    ``per="cell_type"`` columns:
        cell_type, n_cells, n_enrichment_tx, n_total_tx,
        enrichment, enrichment_std
    ``per="cell"`` columns:
        cell_index, cell_type, enrichment, n_total_tx
    """
    return per_cell_fraction_by_type(
        adata, group_labels, wrong_genes.genes,
        value_name="enrichment", min_cells=min_cells, per=per)


def contamination_to_marker_ratio(
    adata: AnnData,
    group_labels: np.ndarray,
    markers: MarkerSet,
    wrong_genes: WrongGeneSet,
    *,
    min_cells: int = 50,
    per: str = "cell_type",
) -> pd.DataFrame:
    """Per-cell ratio of wrong-gene tx to own-marker tx, over cells with
    marker tx > 0, averaged per type. Reveals methods that inflate marker and
    contamination counts proportionally.

    ``per="cell_type"`` columns:
        cell_type, n_cells, n_cells_with_denominator, n_numerator_tx,
        n_denominator_tx, ratio, ratio_std
    ``per="cell"`` columns:
        cell_index, cell_type, ratio
    """
    return per_cell_ratio_by_type(
        adata, group_labels, wrong_genes.genes, markers.genes,
        min_cells=min_cells, per=per)
