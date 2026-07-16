# scref

**sc**RNAseq-**ref**erenced contamination, marker, concordance, and spatial-bleed
analysis for spatial transcriptomics.

`scref` compares a spatial segmentation against a single-cell RNA-seq reference.
It builds cell-type markers, "wrong-gene" contamination signatures, a cell-type
classifier, and a pseudobulk profile **once** from a reference, then scores any
spatial dataset against them. Because every artifact is built from the same
reference in the same gene space, the metrics are mutually consistent: the
wrong-gene set behind the contamination metric is byte-identical to the one
driving the spatial bleed scan, the classifier types cells in the same panel the
markers were ranked in, and so on.

The core pattern: **build one `scRNAseqReference`, score many queries against
it.** This makes `scref` well suited to benchmarking multiple segmentation
methods (e.g. MM Segmentation, Proseg, Segger) on a shared, fixed gene panel,
where every method must be scored against identical marker/wrong-gene
definitions and comparable per-cell denominators.

---

## Install

```bash
pip install -e .            # from the package root
```

Requires `numpy`, `pandas`, `scipy`, `scikit-learn`, `scanpy`, `anndata`.
Marker ranking uses `scanpy.tl.rank_genes_groups`; the spatial bleed module uses
`scipy.spatial.cKDTree`.

Python ≥ 3.9.

---

## Concepts

`scref` distinguishes four artifacts, all built from the reference and all keyed
on the same cell-type vocabulary and gene universe:

- **Markers** — per-cell-type marker gene sets, ranked by `rank_genes_groups`
  (Wilcoxon) with a top-N + log-fold-change floor rule, restricted to the panel,
  with an optional literature fallback for types that yield too few automatic
  markers.
- **Wrong-genes** — per-cell-type *contamination signatures*: genes that are
  strongly expressed in **other** types but not in the target type. Transcripts
  of a type's wrong-genes appearing inside its cells indicate contamination or
  bleed.
- **Classifier** — a `StandardScaler` + `LogisticRegressionCV` cell-type
  classifier fit on the reference in the panel gene space. Used to type query
  cells when you don't supply labels.
- **Pseudobulk** — per-type mean expression + fraction-expressing on
  log1p-normalized counts, captured at build time so a cached reference can
  compute DE concordance against new queries without the original AnnData.

Two ideas make cross-method scoring honest:

**Fixed-panel scoring (`pad_query`).** When you build the reference on a fixed
panel and score several methods against it, a query whose own gene set is
missing some panel genes is **zero-filled** back to the reference's gene
universe before scoring (`pad_query=True`, the default). This puts every method
on the same gene panel, so per-cell denominators and pseudobulk normalization
totals are comparable across methods. Padding only ever adds all-zero columns
for genes the query lacks — for a method that dropped genuinely-empty columns
(e.g. Segger), this reconstructs the true matrix rather than fabricating counts.
Pass `pad_query=False` to score strictly on the query's own genes, or
`score_panel=[...]` to reindex onto an explicit panel.

**Fingerprinting.** Every reference carries a stable `fingerprint` hashing its
build params, gene universe, and marker/wrong-gene contents. Two references that
would score identically hash equal regardless of how they were built; two that
differ in any build param (including the classifier seed, when a classifier was
fit) hash differently. `save`/`load` use it to guard a cache against silent
parameter drift.

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
enrichment = ref.contamination_enrichment(spatial_adata)             # uses ref classifier
ratio      = ref.contamination_to_marker_ratio(spatial_adata, group_by="my_labels")
mark_enr   = ref.marker_enrichment(spatial_adata)
spec       = ref.marker_specificity(spatial_adata)
deconc     = ref.de_concordance(spatial_adata)

# Persist and reload (fingerprint guards against silent param drift)
ref.save("myref.pkl")
ref = scref.scRNAseqReference.load("myref.pkl")
```

Every metric returns a tidy long-format `DataFrame` (one row per cell type, per
(cell type, marker) pair, or per cell-type pair), so results across methods and
cohorts concatenate and plot directly.

### scRNAseq self-baseline

Scoring the reference against its **own** labels gives a lower bound on each
metric — the contamination/bleed you'd measure even with perfect segmentation,
attributable to the reference biology and the panel rather than the method.
Subtract or overlay it to isolate the segmentation-induced signal:

```python
baseline = ref.contamination_enrichment(scrna_adata, group_by="cell_type")
```

---

## `group_by` and typing

Every metric takes `group_by`, which controls how query cells are typed:

- `group_by=None` → cells are typed by the reference's own classifier
  (`ref.predict`). The classifier must have been fit (`fit_classifier=True`, the
  default at build time).
- `group_by="<obs column>"` → use existing labels from `adata.obs`.

---

## API

### `scRNAseqReference.from_anndata(adata, *, annotation_col, panel_genes=None, ...)`

Build a reference. Resolves counts into gene-symbol space once, optionally
restricts to `panel_genes`, then builds markers, wrong-genes, the classifier,
and the pseudobulk in that shared universe. Set `panel_genes` to the
intersection of the reference and query panels when you know the query up front;
leave it `None` to build against the reference's full gene set and rely on
per-call gene handling.

Key parameters:

| group | parameters |
|---|---|
| **panel** | `panel_genes` (restrict gene universe), `raw` (`"auto"`/`"always"`/`"never"`), `feature_name_col` |
| **markers** | `lfc_floor`, `score_quantile`, `pval_cutoff`, `max_markers`, `min_markers_before_fallback`, `min_cells_for_ranking`, `min_ref_cells`, `literature_markers` |
| **wrong-genes** | `wrong_lfc_threshold`, `frac_expr_in_target_floor`, `build_wrong_genes` |
| **classifier** | `fit_classifier`, `min_cells_per_class`, `classifier_cv`, `classifier_n_jobs`, `classifier_random_state` |
| **concordance** | `store_pseudobulk`, `pseudobulk_min_cells` |

Marker defaults were validated against a 17-class harmonized vocabulary across
many cancer datasets. To retarget the literature fallback to a different
vocabulary, pass your own `literature_markers` dict (`{cell_type: [genes]}`); a
default immune/stromal/epithelial panel is used otherwise.

### Metric methods

All take `(adata, *, group_by=None, ..., score_panel=None, pad_query=True)` and
return a tidy `DataFrame`.

| method | measures | key columns |
|---|---|---|
| `contamination_enrichment` | per-cell wrong-gene tx rate, averaged per type | `cell_type, n_cells, n_enrichment_tx, n_total_tx, enrichment, enrichment_std` |
| `marker_enrichment` | per-cell **own-marker** tx rate, averaged per type (symmetric to contamination) | `cell_type, n_cells, n_enrichment_tx, n_total_tx, enrichment, enrichment_std` |
| `contamination_to_marker_ratio` | per-cell wrong-tx / own-marker-tx, per type | `cell_type, n_cells, n_cells_with_denominator, n_numerator_tx, n_denominator_tx, ratio, ratio_std` |
| `marker_specificity` | fraction of each marker's tx landing in the right type | `cell_type, marker, total_counts, correct_counts, specificity` |
| `de_concordance` | Spearman of reference vs query DE per cell-type pair | `type_a, type_b, pair, n_cells_a, n_cells_b, n_genes, spearman` |

`contamination_to_marker_ratio` reveals methods that inflate marker and
contamination counts proportionally (a high raw contamination rate is less
concerning if own-marker signal scales with it).

**Per-cell output.** `contamination_enrichment`, `marker_enrichment`, and
`contamination_to_marker_ratio` accept `per="cell"` (default `"cell_type"`) to
return one row per cell instead of a per-type aggregate:

```python
per_cell = ref.contamination_enrichment(spatial_adata, per="cell")
# columns: cell_index, cell_type, enrichment, n_total_tx
```

**Floors.** `contamination_*` and `marker_enrichment` take a `min_cells` floor;
`marker_specificity` takes `min_total_counts`; `de_concordance` takes `min_cells`
and `min_expr_frac`. Types (or genes) below the floor are dropped from the
result.

### Spatial bleed

Molecular bleed is a transcript generated by a nearby cell of type `T_source`
ending up inside a cell of type `T_sink` (e.g. via diffusion), measured as a
function of distance to the nearest source-type cell. The **eligible** genes for
a `(source, sink)` pair are `markers[source] ∩ wrong[sink]` — source markers
that are also contamination signatures for the sink — drawn from the same
reference, so bleed is consistent with the contamination metrics.

| method | measures |
|---|---|
| `bleed_scan(adata, *, group_by=None, max_dist=40.0, min_sink_cells=50, coord_keys=("x_centroid","y_centroid"), ...)` | per-(source, sink, sink-cell) eligible-gene counts vs distance to nearest source cell |
| `bleed_null_comparison(adata, *, n_null_draws=5, n_expr_bins=20, seed=42, r_grid=None, max_dist=40.0, ...)` | true vs expression-matched-null cumulative bleed curves |

`bleed_null_comparison` runs the scan and its null draws on the **same** scored
matrix, so a genuine distance-dependent bleed signal can be separated from what
an expression-matched null would produce by chance.

`bleed_scan` requires spatial coordinates in `obs` (default columns
`x_centroid`, `y_centroid`; override with `coord_keys`). Distances are in the
coordinate units (microns for Xenium). Aggregations on a scan frame live in
`scref.spatial`:

```python
scan   = ref.bleed_scan(spatial_adata, max_dist=20.0)
radius = scref.spatial.bleed_at_radius(scan, r=10.0, direction="sink")
curve  = scref.spatial.bleed_cumulative(scan, max_dist=20.0)   # B(r) curve
```

### Predict / inspect

```python
labels = ref.predict(spatial_adata)              # np.ndarray of cell-type calls
proba  = ref.predict_proba(spatial_adata)        # DataFrame, one column per class
ref.cell_types          # the reference's annotation vocabulary
ref.gene_universe       # the panel everything was built in
ref.markers             # MarkerSet
ref.wrong_genes         # WrongGeneSet
ref.fingerprint         # stable hash of the build params (see Concepts)
```

`ref.markers.sources` records, per cell type, how its markers were obtained
(e.g. `"auto (37)"` or `"fallback (auto=2, lit=11)"`) for diagnostics.

### Save / load and the fingerprint

`ref.save(path)` writes a versioned envelope via an atomic write (temp sibling +
`os.replace`, so a crash mid-write can't leave a truncated pickle). The envelope
wraps `{format_version, scref_version, fingerprint, annotation_col, cell_types,
reference}`, letting a caller sniff the fingerprint and cell types without
unpickling the estimators. Parent directories are **not** created automatically
— the caller owns the cache layout.

`load(path, expect_fingerprint=...)` re-reads it and raises `ScrefError` if the
file isn't a scref envelope, if its `format_version` is newer than this code
understands, or — when `expect_fingerprint` is given — if the loaded reference's
fingerprint doesn't match. Pass the fingerprint you keyed the cache on so a
key/value mismatch fails loudly instead of silently mis-scoring.

The fingerprint folds in the classifier seed **only** when a classifier was fit,
so references built without a classifier keep a stable fingerprint while two
classifier-bearing references differing only in seed hash differently.

---

## Preprocessing helpers (`scref.preprocessing`)

The most bug-prone contract in a spatial pipeline is *where the counts live and
what id space the `var_names` are in*. `counts_in_symbol_space` is the single
place that resolves it, so every downstream computation operates on raw counts
indexed by gene symbol regardless of input layout.

| function | purpose |
|---|---|
| `counts_in_symbol_space(adata, *, raw="auto", feature_name_col="feature_name")` | resolve raw counts into gene-symbol space (counts from `.raw` when present under `"auto"`; duplicate symbols dropped, first kept) |
| `normalize_log1p(adata, target_sum=1e4)` | total-count normalize + log1p |
| `intersect_genes(a, b)` | shared genes between two AnnData |
| `reindex_to_panel(adata, panel, fill=0.0)` | reindex to a gene panel, zero-filling absent genes (preserves sparsity) |

`reindex_to_panel` is what backs `pad_query`.

---

## Exceptions (`scref.exceptions`)

`ScrefError` (base), `AnnotationColumnError`, `GeneSpaceError`, `NotFittedError`.
`NotFittedError` is raised when you request an artifact the reference wasn't
built with (e.g. `de_concordance` without `store_pseudobulk=True`, or the
classifier when `fit_classifier=False`).

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
    markers.py        marker-enrichment / marker-specificity kernels
    concordance.py    pseudobulk + DE-concordance kernel
    _shared.py        per-cell aggregation helpers
  types.py            MarkerSet, WrongGeneSet, Pseudobulk
  exceptions.py
```

Metric kernels in `scref/metrics/` are stateless functions operating on plain
arrays/dicts; the `scRNAseqReference` methods are thin wrappers that supply the
reference's own markers/wrong-genes/labels and handle panel preparation. The
kernels are importable directly (e.g. `scref.spatial.bleed_scan`) for standalone
use.

---

## Notes

- **Reproducibility.** Pin `classifier_random_state` when building a reference
  you intend to cache; it folds into the fingerprint so the cache identity is
  honest.
- **Column schemas.** Methods return scref-native column names. Pipelines that
  expect legacy names should rename at the boundary.
- **No pipeline concerns.** The reference holds no notion of cancer type,
  method, version, or on-disk caching — those belong to the calling pipeline,
  which attaches them to the returned frames.
- **Version.** 0.1.0.
