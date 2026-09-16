"""Self-supervised learning objectives."""

from .vicreg import VICRegLoss, VICRegTerms, effective_rank, off_diagonal

__all__ = ["VICRegLoss", "VICRegTerms", "effective_rank", "off_diagonal"]
