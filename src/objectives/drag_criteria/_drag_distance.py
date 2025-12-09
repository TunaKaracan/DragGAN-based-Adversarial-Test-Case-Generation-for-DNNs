from typing import Any

import torch
from torch import Tensor

from ._drag_criterion import DragCriterion


class DragDistance(DragCriterion):
	"""An objective function that allows to optimize for the length of drag operations."""

	_name: str = "DragD"

	def __init__(
		self,
		inverse: bool = False,
	) -> None:
		"""
		Initialize the DragDistance objective.

		:param inverse: Whether the measure should be inverted.
		"""

		super().__init__(inverse=inverse, allow_batched=True)

	def evaluate(self, *, handles: Tensor, targets: Tensor, img_resolution: int, **_: Any) -> list[float]:
		"""
		Calculate the total drag distance relative to maximum possible drag distance.

		:param handles: The handle coordinates of the operations.
		:param targets: The target coordinates of the operations.
		:param img_resolution: Image resolution of the generator to normalize the distance.
		:param _: Unused kwargs.
		:returns: The value in range [0, 1].
		"""

		diff = (targets - handles).to(torch.float)
		distances = torch.norm(diff, dim=-1)
		result = distances.sum(dim=1) / (4 * (img_resolution ** 2 * 2) ** (1 / 2))

		partial = (-1) ** (2 - self._inverse.real) * result + self._inverse.real

		results: list[float] = partial.tolist()
		return results