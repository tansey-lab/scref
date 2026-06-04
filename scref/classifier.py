"""Logistic-regression cell-type classifier fit on a scRNAseq reference.

Wraps a ``StandardScaler(with_mean=False)`` + ``LogisticRegressionCV`` pair,
trained on log1p-normalized expression restricted to a shared gene set. The
fitted object is plain-old-data (sklearn estimators + a gene array) so it
pickles cleanly into a caller-owned cache.

``predict_proba`` is part of the public surface: posterior confidence at a
matched transcript load is a useful "is this cell real" diagnostic.
"""

from __future__ import annotations

import sys
from typing import Dict, Optional

import numpy as np
import scanpy as sc
from anndata import AnnData
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from .exceptions import GeneSpaceError, NotFittedError

DEFAULT_MIN_CELLS_PER_CLASS = 50


def _normalize_to(adata: AnnData, genes: np.ndarray) -> np.ndarray:
    """log1p-normalized dense matrix on the gene axis ``genes`` (that order).

    Genes the query lacks are zero-filled FIRST, then normalization is computed
    over the full ``genes`` axis — so the per-cell total (the normalize_total
    denominator) is taken over the same gene universe the classifier was
    trained on. A gene absent from the query contributes 0 to that total,
    exactly as if the query had measured it as zero. (Normalizing over the
    present-only subset instead would use a different denominator than training
    and put the scaled features on a different footing than the model learned.)
    """
    present = [g for g in genes if g in set(adata.var_names)]
    sub = adata[:, present].copy()
    Xsub = sub.X
    if hasattr(Xsub, "toarray"):
        Xsub = Xsub.toarray()
    Xsub = np.asarray(Xsub, dtype=float)

    # Zero-fill into the full training gene order BEFORE normalizing.
    col = {g: i for i, g in enumerate(present)}
    full = np.zeros((adata.n_obs, len(genes)), dtype=float)
    for j, g in enumerate(genes):
        if g in col:
            full[:, j] = Xsub[:, col[g]]

    # normalize_total + log1p over the full panel (missing genes are zeros).
    full_ad = AnnData(X=full)
    sc.pp.normalize_total(full_ad, target_sum=1e4)
    sc.pp.log1p(full_ad)
    out = full_ad.X
    if hasattr(out, "toarray"):
        out = out.toarray()
    return np.asarray(out, dtype=float)


class CellTypeClassifier:
    """L2-penalized logistic-regression classifier over cell-type labels."""

    def __init__(self):
        self.model: Optional[LogisticRegressionCV] = None
        self.scaler: Optional[StandardScaler] = None
        self.genes: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.dropped_classes: Dict[str, int] = {}
        self.min_cells_per_class: int = DEFAULT_MIN_CELLS_PER_CLASS
        self.random_state: Optional[int] = None

    @classmethod
    def fit(
        cls,
        counts_symbol_adata: AnnData,
        *,
        annotation_col: str,
        genes: Optional[np.ndarray] = None,
        min_cells_per_class: int = DEFAULT_MIN_CELLS_PER_CLASS,
        cv: int = 3,
        n_jobs: int = 1,
        max_iter: int = 1000,
        random_state: Optional[int] = 0,
        verbose: bool = True,
    ) -> "CellTypeClassifier":
        """Fit on a reference. Classes with fewer than ``min_cells_per_class``
        cells are dropped before fitting (no head for them), mirroring the
        marker-side min-cells floor.

        ``genes`` restricts the feature space (e.g. to the spatial panel);
        None uses the reference's full gene set.

        ``random_state`` pins the LogisticRegressionCV fit so a reference built
        twice with the same params is reproducible. The default int ``cv`` uses
        non-shuffled StratifiedKFold (already deterministic), so the seed mainly
        fixes solver-internal tie-breaking; record it in any cache key derived
        from this classifier so the identity is honest. ``None`` leaves the fit
        unseeded (not recommended for cached references).
        """
        self = cls()
        self.min_cells_per_class = min_cells_per_class
        self.random_state = random_state

        ad = counts_symbol_adata[
            ~counts_symbol_adata.obs[annotation_col].isna()
        ].copy()
        if genes is None:
            genes = np.array(list(ad.var_names))
        else:
            genes = np.array([g for g in genes if g in set(ad.var_names)])
            if len(genes) == 0:
                raise GeneSpaceError(
                    "no overlap between requested classifier genes and the "
                    "reference gene space")
        self.genes = genes

        class_counts = ad.obs[annotation_col].value_counts()
        keep = class_counts.index[class_counts >= min_cells_per_class]
        self.dropped_classes = {
            str(c): int(n) for c, n in class_counts.items()
            if n < min_cells_per_class
        }
        if self.dropped_classes and verbose:
            print(f"[scref.classifier] dropping classes < "
                  f"{min_cells_per_class} cells: "
                  f"{sorted(self.dropped_classes.items(), key=lambda kv: kv[1])}",
                  file=sys.stderr)
        ad = ad[ad.obs[annotation_col].isin(keep)].copy()
        if ad.obs[annotation_col].nunique() < 2:
            raise GeneSpaceError(
                "fewer than 2 classes remain after the min_cells_per_class "
                "filter; lower the threshold or inspect the reference")

        X = _normalize_to(ad, genes)
        self.scaler = StandardScaler(with_mean=False)
        Xs = self.scaler.fit_transform(X)
        y = ad.obs[annotation_col].values

        self.model = LogisticRegressionCV(
            penalty="l2", cv=cv, n_jobs=n_jobs, solver="lbfgs",
            max_iter=max_iter, random_state=random_state)
        self.model.fit(X=Xs, y=y)
        self.classes_ = self.model.classes_
        return self

    def _features(self, adata: AnnData) -> np.ndarray:
        if self.model is None:
            raise NotFittedError("classifier has not been fit")
        X = _normalize_to(adata, self.genes)
        return self.scaler.transform(X)

    def predict(self, adata: AnnData) -> np.ndarray:
        return self.model.predict(self._features(adata))

    def predict_proba(self, adata: AnnData):
        """(n_obs, n_classes) posterior probabilities. Column order is
        ``self.classes_``."""
        return self.model.predict_proba(self._features(adata))