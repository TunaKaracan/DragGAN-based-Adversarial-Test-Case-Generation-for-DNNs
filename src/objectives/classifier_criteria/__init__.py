"""A collection of criteria used for classification tasks."""

from ._adversarial_distance import AdversarialDistance
from ._binary_change import BinaryChange

__all__ = [
    "AdversarialDistance",
    "BinaryChange"
]
