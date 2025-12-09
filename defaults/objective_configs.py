"""A collection of metric selections for specific objectives."""

from src.objectives.classifier_criteria import AdversarialDistance
from src.objectives.classifier_criteria import BinaryChange

from src.objectives.image_criteria import MatrixDistance

from src.objectives.drag_criteria import DragDistance

"""
### Adversarial Testing:
The objective is to find inputs that induce misbehavior in the SUT, while exhibiting minimal changes to the original reference.

DYNAMIC_ADVERSARIAL_TESTING: Allows for a changing target in the optimization, i.e the adversarial class can change.
MULTI_ATTRIBUTE_ADVERSARIAL_TESTING: Same as DYNAMIC_ADVERSARIAL_TESTING but the model tested outputs multiple binary attributes.
"""

DYNAMIC_ADVERSARIAL_TESTING = [
    AdversarialDistance(),
    MatrixDistance(),
    DragDistance()
]

MULTI_ATTRIBUTE_ADVERSARIAL_TESTING = [
    BinaryChange(target_logit=31, flip_sign=True, is_logit=True),
    MatrixDistance(),
    DragDistance()
]

MULTI_ATTRIBUTE_ADVERSARIAL_TESTING2 = [
    BinaryChange(target_logit=31, flip_sign=True, is_logit=False),
    MatrixDistance(),
    DragDistance()
]
