"""The central reference object.

`scRNAseqReference` is built once from a scRNAseq AnnData and thereafter scores
any query AnnData (typically spatial) against the reference's markers,
wrong-genes, and cell-type classifier. It holds no notion of cancer, method,
version, or on-disk caching: those belong to the calling pipeline. It is
pickle-able so a caller can cache a built reference and reload it (including
for DE concordance, via the stored reference pseudobulk) without the original
reference AnnData.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from typing import Dict, Optional, Sequence, Set, Union

import numpy as np
import pandas as pd
from anndata import AnnData

from . import markers as _markers_mod
from . import wronggenes as _wrong_mod
from .classifier import CellTypeClassifier, DEFAULT_MIN_CELLS_PER_CLASS
from .exceptions import AnnotationColumnError, NotFittedError
from .metrics import concordance as _concordance
from .metrics import contamination as _contamination
from .metrics import markers as _marker_metrics
from . import spatial as _spatial
from .preprocessing import counts_in_symbol_space, reindex_to_panel as _reindex_to_panel
from .types import MarkerSet, WrongGeneSet

# Bump when the pickled state layout changes incompatibly (e.g. an attribute is
# renamed/removed such that an old pickle would mis-score under new code). `load`
# refuses envelopes whose format version exceeds what this code understands.
SAVE_FORMAT_VERSION = 1


class scRNAseqReference:
    """A fitted scRNAseq reference for scoring spatial (or other) query data.

    Build with :meth:`from_anndata`. Score with :meth:`contamination_enrichment`,
    :meth:`contamination_to_marker_ratio`, :meth:`marker_enrichment`,
    :meth:`marker_specificity`, and :meth:`de_concordance`. All scoring methods
    accept ``group_by`` (an obs column of pre-assigned labels) and default to
    running the built-in classifier when it is omitted.
    """

    def __init__(
        self,
        *,
        markers: MarkerSet,
        wrong_genes: Optional[WrongGeneSet],
        classifier: Optional[CellTypeClassifier],
        reference_pseudobulk: Optional[_concordance.Pseudobulk],
        annotation_col: str,
        gene_universe: np.ndarray,
        build_params: dict,
    ):
        self._markers = markers
        self._wrong_genes = wrong_genes
        self._classifier = classifier
        self._reference_pseudobulk = reference_pseudobulk
        self._annotation_col = annotation_col
        self._gene_universe = np.asarray(gene_universe)
        self._build_params = dict(build_params)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_anndata(
        cls,
        adata: AnnData,
        *,
        annotation_col: str,
        panel_genes: Optional[Sequence[str]] = None,
        raw: str = "auto",
        feature_name_col: str = "feature_name",
        # marker params
        lfc_floor: float = _markers_mod.DEFAULT_LFC_FLOOR,
        score_quantile: float = _markers_mod.DEFAULT_SCORE_QUANTILE,
        pval_cutoff: float = _markers_mod.DEFAULT_PVAL_CUTOFF,
        max_markers: int = _markers_mod.DEFAULT_MAX_MARKERS,
        min_markers_before_fallback: int = (
            _markers_mod.DEFAULT_MIN_MARKERS_BEFORE_FALLBACK),
        min_cells_for_ranking: int = _markers_mod.DEFAULT_MIN_CELLS_FOR_RANKING,
        min_ref_cells: int = _markers_mod.DEFAULT_MIN_REF_CELLS,
        literature_markers: Optional[Dict[str, list]] = None,
        # wrong-gene params
        wrong_lfc_threshold: float = _wrong_mod.DEFAULT_WRONG_LFC,
        frac_expr_in_target_floor: float = (
            _wrong_mod.DEFAULT_FRAC_IN_TARGET_FLOOR),
        # classifier params
        fit_classifier: bool = True,
        min_cells_per_class: int = DEFAULT_MIN_CELLS_PER_CLASS,
        classifier_cv: int = 3,
        classifier_n_jobs: int = 1,
        classifier_random_state: Optional[int] = 0,
        # concordance support
        store_pseudobulk: bool = True,
        pseudobulk_min_cells: int = 50,
        # behaviour
        build_wrong_genes: bool = True,
        verbose: bool = True,
    ) -> "scRNAseqReference":
        """Build a reference from a scRNAseq AnnData.

        Resolves counts into symbol space once, optionally restricts to
        ``panel_genes`` (so markers, wrong-genes, classifier, and pseudobulk
        all live in the gene universe the query will be scored in), then builds
        every artifact.

        Set ``panel_genes`` to the intersection of the reference and the query
        panel when you know the query up front; leave it None to build against
        the reference's full gene set and rely on per-call gene intersection.
        """
        if annotation_col not in adata.obs.columns:
            raise AnnotationColumnError(
                f"annotation_col {annotation_col!r} not in obs "
                f"({list(adata.obs.columns)})")

        counts = counts_in_symbol_space(
            adata, raw=raw, feature_name_col=feature_name_col)

        if panel_genes is not None:
            panel_set: Optional[Set[str]] = set(map(str, panel_genes))
            in_var = [g for g in counts.var_names if g in panel_set]
            counts = counts[:, in_var].copy()
        else:
            panel_set = None

        gene_universe = np.array(list(counts.var_names))

        markers = _markers_mod.build_markers(
            counts, annotation_col=annotation_col,
            lfc_floor=lfc_floor, score_quantile=score_quantile,
            pval_cutoff=pval_cutoff, max_markers=max_markers,
            min_markers_before_fallback=min_markers_before_fallback,
            min_cells_for_ranking=min_cells_for_ranking,
            min_ref_cells=min_ref_cells,
            panel_genes=panel_set, literature_markers=literature_markers,
            verbose=verbose)

        wrong_genes = None
        if build_wrong_genes:
            wrong_genes = _wrong_mod.build_wrong_genes(
                counts, markers, annotation_col=annotation_col,
                wrong_lfc_threshold=wrong_lfc_threshold,
                frac_expr_in_target_floor=frac_expr_in_target_floor)

        classifier = None
        if fit_classifier:
            classifier = CellTypeClassifier.fit(
                counts, annotation_col=annotation_col,
                genes=gene_universe,
                min_cells_per_class=min_cells_per_class,
                cv=classifier_cv, n_jobs=classifier_n_jobs,
                random_state=classifier_random_state, verbose=verbose)

        reference_pseudobulk = None
        if store_pseudobulk:
            reference_pseudobulk = _concordance.build_pseudobulk(
                counts, group_col=annotation_col, genes=gene_universe,
                min_cells=pseudobulk_min_cells)

        build_params = dict(
            annotation_col=annotation_col,
            lfc_floor=lfc_floor, score_quantile=score_quantile,
            pval_cutoff=pval_cutoff, max_markers=max_markers,
            min_markers_before_fallback=min_markers_before_fallback,
            min_cells_for_ranking=min_cells_for_ranking,
            min_ref_cells=min_ref_cells,
            wrong_lfc_threshold=wrong_lfc_threshold,
            frac_expr_in_target_floor=frac_expr_in_target_floor,
            min_cells_per_class=min_cells_per_class,
        )
        # Record the classifier seed in build_params (hence the fingerprint)
        # ONLY when a classifier was actually fit — so references built without
        # a classifier keep their existing fingerprint, and two classifier-
        # bearing references that differ only in seed hash differently.
        if fit_classifier:
            build_params["classifier_random_state"] = classifier_random_state

        return cls(
            markers=markers, wrong_genes=wrong_genes, classifier=classifier,
            reference_pseudobulk=reference_pseudobulk,
            annotation_col=annotation_col, gene_universe=gene_universe,
            build_params=build_params)

    # ------------------------------------------------------------- properties
    @property
    def markers(self) -> MarkerSet:
        return self._markers

    @property
    def wrong_genes(self) -> WrongGeneSet:
        if self._wrong_genes is None:
            raise NotFittedError("wrong genes were not built for this reference")
        return self._wrong_genes

    @property
    def classifier(self) -> CellTypeClassifier:
        if self._classifier is None:
            raise NotFittedError("no classifier was fit for this reference")
        return self._classifier

    @property
    def annotation_col(self) -> str:
        return self._annotation_col

    @property
    def cell_types(self) -> list:
        return self._markers.cell_types

    @property
    def gene_universe(self) -> np.ndarray:
        return self._gene_universe

    @property
    def fingerprint(self) -> str:
        """Stable hash of the reference's identity for a caller's cache key.

        Combines the build params, gene universe, and the marker / wrong-gene
        fingerprints, so two references that would score identically hash
        equal regardless of how they were constructed."""
        parts = [
            repr(sorted(self._build_params.items())),
            ",".join(map(str, sorted(self._gene_universe.tolist()))),
            self._markers.fingerprint,
            self._wrong_genes.fingerprint if self._wrong_genes else "none",
        ]
        blob = "||".join(parts).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()[:16]

    # ------------------------------------------------------------ persistence
    def save(self, path: Union[str, os.PathLike], *,
             protocol: int = pickle.HIGHEST_PROTOCOL) -> str:
        """Serialize this reference to ``path`` (a single pickle file).

        The on-disk file is a small envelope wrapping the object:
        ``{format_version, scref_version, fingerprint, annotation_col,
        cell_types, reference}``. The metadata lets :meth:`load` reject a
        pickle written by an incompatible future format and lets a caller
        sniff the fingerprint / cell types without unpickling the estimators.

        Returns the path written. Parent directories are NOT created
        automatically — the caller owns the cache layout.
        """
        from . import __version__ as _scref_version
        envelope = {
            "format_version": SAVE_FORMAT_VERSION,
            "scref_version": _scref_version,
            "fingerprint": self.fingerprint,
            "annotation_col": self._annotation_col,
            "cell_types": self.cell_types,
            "reference": self,
        }
        path = os.fspath(path)
        # Write to a temp sibling then atomically replace, so a crash mid-write
        # can't leave a truncated pickle that later reads as a corrupt cache.
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "wb") as io:
            pickle.dump(envelope, io, protocol=protocol)
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: Union[str, os.PathLike], *,
             expect_fingerprint: Optional[str] = None) -> "scRNAseqReference":
        """Load a reference written by :meth:`save`.

        Raises ``ScrefError`` if the file is not a recognized scref envelope,
        if its ``format_version`` is newer than this code understands, or if
        the unpickled object is not a ``scRNAseqReference``.

        ``expect_fingerprint`` (optional) guards against loading the wrong
        cached artifact: if given and the loaded reference's fingerprint does
        not match, a ``ScrefError`` is raised. Pass the fingerprint the caller
        keyed its cache on to make a key/value mismatch fail loudly.
        """
        from .exceptions import ScrefError
        path = os.fspath(path)
        with open(path, "rb") as io:
            envelope = pickle.load(io)

        if not (isinstance(envelope, dict)
                and envelope.get("format_version") is not None
                and "reference" in envelope):
            raise ScrefError(
                f"{path!r} is not a scref reference envelope (was it written "
                f"by scRNAseqReference.save?)")

        fmt = envelope["format_version"]
        if fmt > SAVE_FORMAT_VERSION:
            raise ScrefError(
                f"{path!r} was written with save-format v{fmt}, but this scref "
                f"only understands up to v{SAVE_FORMAT_VERSION}; upgrade scref "
                f"to load it.")

        ref = envelope["reference"]
        if not isinstance(ref, cls):
            raise ScrefError(
                f"{path!r} unpickled to {type(ref).__name__}, not "
                f"scRNAseqReference")

        if (expect_fingerprint is not None
                and ref.fingerprint != expect_fingerprint):
            raise ScrefError(
                f"fingerprint mismatch loading {path!r}: expected "
                f"{expect_fingerprint!r}, got {ref.fingerprint!r}. The cached "
                f"reference does not match the requested build params.")

        return ref

    # --------------------------------------------------------------- labeling
    def predict(self, adata: AnnData) -> np.ndarray:
        """Predicted cell-type labels for each cell in ``adata``."""
        return self.classifier.predict(adata)

    def predict_proba(self, adata: AnnData) -> pd.DataFrame:
        """Posterior class probabilities as a DataFrame (columns = classes)."""
        proba = self.classifier.predict_proba(adata)
        return pd.DataFrame(proba, columns=list(self.classifier.classes_))

    def _resolve_labels(self, adata: AnnData,
                        group_by: Optional[str]) -> np.ndarray:
        if group_by is None:
            return self.predict(adata)
        if group_by not in adata.obs.columns:
            raise AnnotationColumnError(
                f"group_by={group_by!r} not in query obs "
                f"({list(adata.obs.columns)})")
        return np.asarray(adata.obs[group_by].values)

    def _prepare_query(self, adata: AnnData, group_by: Optional[str],
                       score_panel: Optional[Sequence[str]],
                       pad_query: bool = True):
        """Resolve per-cell labels, then put the query on a fixed gene panel so
        every method is scored against the SAME reference universe.

        Panel resolution:
          - explicit ``score_panel``        -> reindex onto it (caller override)
          - else ``pad_query`` (default)    -> reindex onto the reference's
                                               gene_universe, zero-filling any
                                               gene the query lacks
          - else                            -> score the query as-is

        Labels are resolved on the ORIGINAL adata first (cell order is preserved
        by reindex_to_panel, so labels stay aligned). Padding only ever adds
        all-zero columns for genes the query lacks; for a method that dropped
        genuinely-empty columns (e.g. Segger), this reconstructs the true matrix
        and makes per-cell denominators / normalization totals identical across
        methods.
        """
        labels = self._resolve_labels(adata, group_by)
        if score_panel is not None:
            return _reindex_to_panel(adata, score_panel), labels
        if pad_query and not self._query_matches_universe(adata):
            return _reindex_to_panel(adata, self._gene_universe), labels
        return adata, labels

    def _query_matches_universe(self, adata: AnnData) -> bool:
        """True iff the query's var axis is already exactly the reference
        gene_universe (same set), so padding would be a no-op."""
        return set(map(str, adata.var_names)) == set(map(str,
                                                         self._gene_universe))

    # ----------------------------------------------------------------- metrics
    def contamination_enrichment(
        self, adata: AnnData, *, group_by: Optional[str] = None,
        per: str = "cell_type", min_cells: int = 50,
        score_panel: Optional[Sequence[str]] = None, pad_query: bool = True,
    ) -> pd.DataFrame:
        scored, labels = self._prepare_query(adata, group_by, score_panel,
                                             pad_query)
        return _contamination.contamination_enrichment(
            scored, labels, self.wrong_genes, min_cells=min_cells, per=per)

    def contamination_to_marker_ratio(
        self, adata: AnnData, *, group_by: Optional[str] = None,
        per: str = "cell_type", min_cells: int = 50,
        score_panel: Optional[Sequence[str]] = None, pad_query: bool = True,
    ) -> pd.DataFrame:
        scored, labels = self._prepare_query(adata, group_by, score_panel,
                                             pad_query)
        return _contamination.contamination_to_marker_ratio(
            scored, labels, self.markers, self.wrong_genes,
            min_cells=min_cells, per=per)

    def marker_enrichment(
        self, adata: AnnData, *, group_by: Optional[str] = None,
        per: str = "cell_type", min_cells: int = 50,
        score_panel: Optional[Sequence[str]] = None, pad_query: bool = True,
    ) -> pd.DataFrame:
        scored, labels = self._prepare_query(adata, group_by, score_panel,
                                             pad_query)
        return _marker_metrics.marker_enrichment(
            scored, labels, self.markers, min_cells=min_cells, per=per)

    def marker_specificity(
        self, adata: AnnData, *, group_by: Optional[str] = None,
        min_total_counts: int = 50,
        score_panel: Optional[Sequence[str]] = None, pad_query: bool = True,
    ) -> pd.DataFrame:
        scored, labels = self._prepare_query(adata, group_by, score_panel,
                                             pad_query)
        return _marker_metrics.marker_specificity(
            scored, labels, self.markers, min_total_counts=min_total_counts)

    def de_concordance(
        self, adata: AnnData, *, group_by: Optional[str] = None,
        min_cells: int = 50, min_expr_frac: float = 0.05,
        score_panel: Optional[Sequence[str]] = None, pad_query: bool = True,
    ) -> pd.DataFrame:
        if self._reference_pseudobulk is None:
            raise NotFittedError(
                "reference pseudobulk was not stored; rebuild with "
                "store_pseudobulk=True to use de_concordance")
        scored, labels = self._prepare_query(adata, group_by, score_panel,
                                             pad_query)
        scored = scored.copy()
        scored.obs["_scref_group"] = labels
        return _concordance.de_concordance(
            self._reference_pseudobulk, scored, query_group_col="_scref_group",
            min_cells=min_cells, min_expr_frac=min_expr_frac)

    # ------------------------------------------------------------- spatial bleed
    def _prepare_spatial(self, adata, group_by, score_panel, pad_query,
                         coord_keys):
        """Like _prepare_query, but carries the spatial coordinate columns onto
        the (possibly padded) scored object so the bleed kernel can read them.
        Returns (scored_adata, labels)."""
        scored, labels = self._prepare_query(adata, group_by, score_panel,
                                             pad_query)
        if scored is not adata:
            # reindex_to_panel preserves obs, so the coord columns survive; this
            # check is just to give a clear error if a caller passed coords that
            # aren't in obs at all.
            for c in coord_keys:
                if c not in scored.obs.columns:
                    raise AnnotationColumnError(
                        f"coordinate column {c!r} not in query obs")
        return scored, labels

    def bleed_scan(
        self, adata: AnnData, *, group_by: Optional[str] = None,
        max_dist: float = 40.0, min_sink_cells: int = 50,
        coord_keys: Sequence[str] = _spatial.DEFAULT_COORD_KEYS,
        sink_mask: Optional[np.ndarray] = None,
        score_panel: Optional[Sequence[str]] = None, pad_query: bool = True,
    ) -> pd.DataFrame:
        """Per-(source, sink, sink-cell) molecular-bleed scan against this
        reference's markers and wrong-genes. See ``scref.spatial.bleed_scan``.

        Requires the spatial coordinate columns (``coord_keys``) in obs. Returns
        the long-form scan frame WITHOUT cancer/method/version (the caller's to
        attach). ``sink_cell_row`` indexes into the scored (padded) adata.

        ``sink_mask`` (boolean, aligned to ``adata`` cell order) restricts the
        measured sink cells while keeping the full point pattern as the source
        pool -- see ``scref.spatial.bleed_scan``. ``_prepare_spatial`` preserves
        obs order, so a mask built on the input ``adata`` aligns with ``scored``.
        """
        scored, labels = self._prepare_spatial(adata, group_by, score_panel,
                                               pad_query, coord_keys)
        return _spatial.bleed_scan(
            scored, labels, self.markers.genes, self.wrong_genes.genes,
            max_dist=max_dist, min_sink_cells=min_sink_cells,
            coord_keys=coord_keys, sink_mask=sink_mask)

    def bleed_null_comparison(
        self, adata: AnnData, *, group_by: Optional[str] = None,
        max_dist: float = 40.0, min_sink_cells: int = 50,
        coord_keys: Sequence[str] = _spatial.DEFAULT_COORD_KEYS,
        sink_mask: Optional[np.ndarray] = None,
        n_null_draws: int = 5, n_expr_bins: int = 20, seed: int = 42,
        r_grid=None,
        score_panel: Optional[Sequence[str]] = None, pad_query: bool = True,
    ) -> pd.DataFrame:
        """True-vs-expression-matched-null bleed cumulative curves.

        Runs ``bleed_scan`` and the null draws on the SAME scored matrix, so the
        null's ``sink_cell_row`` indexing aligns with the scan. See
        ``scref.spatial.bleed_null_comparison``.

        ``sink_mask`` is forwarded to the underlying ``bleed_scan``, so the true
        and null curves are measured on the same masked sink population against
        the full source pool. The null's expression-bin matching is computed over
        the full ``scored`` matrix (applied identically to true and null).
        """
        scored, labels = self._prepare_spatial(adata, group_by, score_panel,
                                               pad_query, coord_keys)
        bleed = _spatial.bleed_scan(
            scored, labels, self.markers.genes, self.wrong_genes.genes,
            max_dist=max_dist, min_sink_cells=min_sink_cells,
            coord_keys=coord_keys, sink_mask=sink_mask)
        return _spatial.bleed_null_comparison(
            scored, bleed, self.markers.genes,
            n_null_draws=n_null_draws, n_expr_bins=n_expr_bins, seed=seed,
            r_grid=r_grid, max_dist=max_dist)