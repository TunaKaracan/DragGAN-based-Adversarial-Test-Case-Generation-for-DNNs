from typing import Any, Optional

import torch
from torch import Tensor

from ._classifier_criterion import ClassifierCriterion


class BinaryChange(ClassifierCriterion):
    """Implements a criterion that observes change in binary classification."""

    _name: str = "BinaryChange"
    _precondition_logits: Optional[Tensor]

    def __init__(
        self,
        target_logit: Optional[int] = None,
        flip_sign: bool = False,
        boundary: bool = False,
        is_logit: bool = False,
        inverse: bool = False,
        v_range: Optional[tuple[float, float]] = None,
    ) -> None:
        """
        Initialize the BinaryChange Criterion.

        By default, the change is expected to be a decrease in confidence.

        :param target_logit: Which logit should be observed for change (some classifiers have multiple binary outputs).
        :param flip_sign: Whether the sign should be flipped instead of pure increase / decrease of confidence.
        :param boundary: If boundary should be approached (i.e 0 Logit or 0.5 confidence).
        :param is_logit: Whether the output is logits or probabilities.
        :param inverse: Whether the criterion should be inverted.
        :param v_range: Set a range of the input if applicable, to normalize outputs to [0,1].
        :raises NotImplementedError: If target_logit is None.
        """

        super().__init__(inverse=inverse, allow_batched=True)
        self._target_logit = target_logit
        self._flip_sign = flip_sign
        self._boundary = boundary
        self._is_logit = is_logit
        self._v_range = v_range or (0.0, 1.0)

        if target_logit is None:
            raise NotImplementedError(
                "Target logit must be specified. Or implement handling for returning multiple :)."
            )

    def evaluate(self, *, logits: Tensor, **_: Any) -> list[float]:
        """
        Calculate the change in binary confidence values.

        This function returns normalized values if the v_range was set in initialization.

        :param logits: Logits tensor.
        :param _: Unused kwargs.
        :returns: The value of the change.
        :raises ValueError: If precondition logits are not set before flipping sign.
        """

        logits = logits[:, self._target_logit].detach().clone() if self._target_logit else logits.detach().clone()
        precondition = self._precondition_logits.detach().clone()

        score = torch.zeros_like(logits)

        # Move the probability space from [0, 1] -> [-0.5, 0.5] to center it around 0, similar to logit space.
        if not self._is_logit:
            logits -= 0.5
            precondition -= 0.5

        if self._flip_sign:
            if self._precondition_logits is None:
                raise ValueError("Precondition logits must be set before flipping sign.")
            # Encourage sign flip: -1 if flipped, +1 if not.
            score += torch.sign(logits * precondition) * torch.abs(logits)
        else:
            score += (-1) ** (2 - self._inverse.real) * logits

        # Re-center the score boundary to [0, 1].
        if not self._is_logit:
            score += 0.5

        if self._boundary:
            # Encourage small magnitude (approaching 0).
            score += -torch.abs(logits)

        partial = (score - self._v_range[0]) / (self._v_range[1] - self._v_range[0])
        results: list[float] = partial.tolist()
        return results

    def precondition(self, *, logits: Tensor, **_: Any) -> None:
        """
        Calculate the change in binary confidence values.

        This function returns normalized values if the v_range was set in initialization.

        :param logits: Logits tensor.
        :param _: Unused kwargs.
        """
        self._precondition_logits = logits[:, self._target_logit] if self._target_logit else logits