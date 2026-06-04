"""scref — a reusable interface for scRNAseq-referenced contamination,
marker, and concordance analysis of spatial transcriptomics data.

Build a reference once from a scRNAseq AnnData, then score any query AnnData::

    from scref import scRNAseqReference

    ref = scRNAseqReference.from_anndata(
        scrna_adata, annotation_col="harmonized_annotation",
        panel_genes=spatial_adata.var_names)

    ce  = ref.contamination_enrichment(spatial_adata)            # per cell type
    cec = ref.contamination_enrichment(spatial_adata, per="cell")  # per cell
    cmr = ref.contamination_to_marker_ratio(spatial_adata)
    me  = ref.marker_enrichment(spatial_adata)
    ms  = ref.marker_specificity(spatial_adata)
    dec = ref.de_concordance(spatial_adata)

    # scRNAseq self-baseline: score the reference against its own labels
    ce_ref = ref.contamination_enrichment(scrna_adata,
                                          group_by="harmonized_annotation")
"""

from .reference import scRNAseqReference
from .classifier import CellTypeClassifier
from .types import MarkerSet, WrongGeneSet
from .metrics.concordance import Pseudobulk
from . import spatial
from . import exceptions

__all__ = [
    "scRNAseqReference",
    "CellTypeClassifier",
    "MarkerSet",
    "WrongGeneSet",
    "Pseudobulk",
    "spatial",
    "exceptions",
]

__version__ = "0.1.0"