"""Post-seal evaluation interfaces.

Runtime modules must not import this package because it owns query-vault readers.
"""

from mcss.evaluation.sealed_queries import evaluate_sealed

__all__ = ["evaluate_sealed"]
