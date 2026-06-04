# scref

**sc**RNAseq-**ref**erenced contamination, marker, concordance, and spatial-bleed
analysis for spatial transcriptomics.

`scref` is a package to compare a spatial segmentation against a
single-cell RNA-seq reference: it builds cell-type markers, "wrong-gene"
contamination signatures, a cell-type classifier, and a pseudobulk profile
*once* from a reference, then scores any spatial dataset against them.

The core idea: **build one `scRNAseqReference`, score many queries against it.**

---

## Install

```bash
pip install -e .            # from the package root
```

Requires `numpy`, `pandas`, `scipy`, `scikit-learn`, `anndata`. (Spatial bleed
uses `scipy.spatial.cKDTree`; nothing else is needed for the spatial module.)

---

## Quick start

```python
import scref

# Build a reference from an annotated scRNAseq AnnData. Markers, wrong-genes,
# a logistic-regression classifier, and a pseudobulk are all built here, on the
# (optional) panel you pass -- so every artifact lives in the same gene space
# the spatial query will be scored in.
ref = scref.scRNAseqReference.from_anndata(
    scrna_adata,
    annotation_col="cell_type",
    panel_genes=panel,                 # e.g. spatial panel ∩ scRNAseq; None = all genes
    classifier_random_state=0,         # pin for reproducibility
)

# Score a spatial AnnData. group_by=None types cells with the reference's own
# classifier; pass an obs column to use existing labels instead.
enrichment = ref.contamination_enrichment(spatial_adata)            # uses ref classifier
ratio      = ref.contamination_to_marker_ratio(spatial_adata, group_by="my_labels")
spec       = ref.marker_specificity(spatial_adata)
deconc     = ref.de_concordance(spatial_adata)

# Persist and reload (fingerprint guards against silent param drift)
ref.save("myref.pkl")
ref = scref.scRNAseqReference.load("myref.pkl")
```

Every metric returns a tidy long-format `DataFrame` (one row per cell type,
per (cell type, marker) pair, or per cell-type pair), so results concatenate
and plot directly.

---

## Why one reference object

Markers, wrong-genes, the classifier, and the pseudobulk are all functions of
the *same* reference and the *same* gene panel. Building them together in one
object guarantees they are mutually consistent — the wrong-gene set used by the
contamination metric is byte-identical to the one used by the spatial bleed
scan, the classifier types cells in the same panel the markers were ranked in,
and so on. It also means you pay the build cost once and reuse it across every
method and every metric.

### Fixed-panel scoring (`pad_query`)

When you build the reference on a fixed panel and score several methods against
it, a query whose own gene set is missing some panel genes is **zero-filled**
back to the reference's gene universe before scoring (`pad_query=True`, the
default). This puts every method on the same gene panel, so per-cell
denominators and pseudobulk normalization totals are comparable across methods.
If a method legitimately produced zero transcripts for a gene, zero-filling is
exact, not fabrication. Pass `pad_query=False` to score strictly on the query's
own genes.

### `group_by` and typing

Every metric takes `group_by`:

- `group_by=None` → cells are typed by the reference's own classifier
  (`ref.predict`). The classifier must have been fit (`fit_classifier=True`,
  the default).
- `group_by="<obs column>"` → use existing labels from `adata.obs`.

---

## API

### `scRNAseqReference.from_anndata(adata, *, annotation_col, panel_genes=None, ...)`

Build a reference. Key parameters:

| group | parameters |
|---|---|
| **panel** | `panel_genes` (restrict gene universe), `raw`, `feature_name_col` |
| **markers** | `lfc_floor`, `score_quantile`, `pval_cutoff`, `max_markers`, `min_markers_before_fallback`, `min_ref_cells`, `literature_markers` |
| **wrong-genes** | `wrong_lfc_threshold`, `frac_expr_in_target_floor`, `build_wrong_genes` |
| **classifier** | `fit_classifier`, `min_cells_per_class`, `classifier_cv`, `classifier_n_jobs`, `classifier_random_state` |
| **concordance** | `store_pseudobulk`, `pseudobulk_min_cells` |

### Metric methods

All take `(adata, *, group_by=None, ..., score_panel=None, pad_query=True)`
and return a tidy `DataFrame`.

| method | measures | key columns |
|---|---|---|
| `contamination_enrichment` | per-cell wrong-gene tx rate, averaged per type | `cell_type, n_cells, n_enrichment_tx, n_total_tx, enrichment, enrichment_std` |
| `contamination_to_marker_ratio` | per-cell wrong-tx / own-marker-tx, per type | `cell_type, n_cells, n_cells_with_denominator, n_numerator_tx, n_denominator_tx, ratio, ratio_std` |
| `marker_specificity` | fraction of each marker's tx landing in the right type | `cell_type, marker, total_counts, correct_counts, specificity` |
| `de_concordance` | Spearman of reference vs query DE per cell-type pair | `type_a, type_b, pair, n_cells_a, n_cells_b, n_genes, spearman` |

`contamination_*` and `marker_specificity` take `min_cells` / `min_total_counts`
floors; `de_concordance` takes `min_cells` and `min_expr_frac`.

### Spatial bleed

| method | measures |
|---|---|
| `bleed_scan(adata, *, group_by=None, max_dist=40.0, min_sink_cells=50, coord_keys=("x_centroid","y_centroid"), ...)` | per-(source, sink, sink-cell) eligible-gene counts vs distance to nearest source cell |
| `bleed_null_comparison(adata, *, n_null_draws=5, n_expr_bins=20, seed=42, r_grid=None, max_dist=40.0, ...)` | true vs expression-matched-null cumulative bleed curves |

The "eligible" genes for a (source, sink) pair are `markers[source] ∩
wrong[sink]` — drawn from the same reference, so bleed is consistent with the
contamination metrics. Aggregations on a scan frame live in `scref.spatial`:

```python
scan = ref.bleed_scan(spatial_adata, max_dist=20.0)
import scref
radius = scref.spatial.bleed_at_radius(scan, r=10.0, direction="sink")
curve  = scref.spatial.bleed_cumulative(scan, max_dist=20.0)   # B(r) curve
```

`bleed_scan` requires spatial coordinates in `obs` (default columns
`x_centroid`, `y_centroid`; override with `coord_keys`). Distances are in the
coordinate units (microns for Xenium).

### Predict / inspect

```python
labels = ref.predict(spatial_adata)              # np.ndarray of cell-type calls
proba  = ref.predict_proba(spatial_adata)        # DataFrame, one column per class
ref.cell_types          # the reference's annotation vocabulary
ref.gene_universe       # the panel everything was built in
ref.markers             # MarkerSet
ref.wrong_genes         # WrongGeneSet
ref.fingerprint         # stable hash of the build params (see below)
```

### Save / load and the fingerprint

`ref.save(path)` writes a versioned envelope (atomic write). `load(path,
expect_fingerprint=...)` re-reads it and, if you pass a fingerprint, raises if
the loaded object's build params don't match — so a cached reference can't
silently diverge from the parameters you think produced it. The fingerprint
hashes all marker / wrong-gene / classifier build params, including the
classifier seed (only when a classifier was fit), so two references that differ
in any of those hash differently.

---

## Preprocessing helpers (`scref.preprocessing`)

| function | purpose |
|---|---|
| `counts_in_symbol_space(adata, ...)` | resolve raw counts into gene-symbol space |
| `normalize_log1p(adata, target_sum=1e4)` | total-count normalize + log1p |
| `intersect_genes(a, b)` | shared genes between two AnnData |
| `reindex_to_panel(adata, panel, fill=0.0)` | reindex to a gene panel, zero-filling absent genes (preserves sparsity) |

`reindex_to_panel` is what backs `pad_query`.

---

## Exceptions (`scref.exceptions`)

`ScrefError` (base), `AnnotationColumnError`, `GeneSpaceError`, `NotFittedError`.

---

## Package layout

```
scref/
  reference.py        scRNAseqReference — the central object
  markers.py          build_markers
  wronggenes.py       build_wrong_genes
  classifier.py       CellTypeClassifier (StandardScaler + LogisticRegressionCV)
  preprocessing.py    counts_in_symbol_space, normalize_log1p, reindex_to_panel, ...
  spatial.py          bleed_scan, bleed_cumulative, bleed_at_radius, bleed_null_comparison
  metrics/
    contamination.py  enrichment / wrong-to-marker-ratio kernels
    markers.py        marker-specificity kernel
    concordance.py    pseudobulk + DE-concordance kernel
    _shared.py        per-cell aggregation helpers
  types.py            MarkerSet, WrongGeneSet
  exceptions.py
```

Metric kernels in `scref/metrics/` are stateless functions operating on plain
arrays/dicts; the `scRNAseqReference` methods are thin wrappers that supply the
reference's own markers/wrong-genes/labels and handle panel preparation.

---

## Notes

- **Reproducibility.** Pin `classifier_random_state` when building a reference
  you intend to cache; it folds into the fingerprint so the cache identity is
  honest.
- **Column schemas.** Methods return scref-native column names. Pipelines that
  expect legacy names should rename at the boundary.
- **Version.** 0.1.0.
