"""Exceptions raised by scref."""

from __future__ import annotations


class ScrefError(Exception):
    """Base class for all scref errors."""


class AnnotationColumnError(ScrefError):
    """The requested annotation column is missing from an AnnData's obs."""


class GeneSpaceError(ScrefError):
    """A query AnnData shares too few genes with the reference, or a required
    gene-symbol resolution failed."""


class NotFittedError(ScrefError):
    """A scoring method was called on a reference whose required artifact
    (markers, wrong-genes, classifier, or pseudobulk) was not built."""
