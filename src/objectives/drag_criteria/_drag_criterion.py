from abc import abstractmethod
from typing import Any, Union

from torch import Tensor

from .._criterion import Criterion


class DragCriterion(Criterion):
	"""A criterion that only considers dragging operations."""

	def __init__(self, inverse: bool = False, allow_batched: bool = False) -> None:
		"""
		Initialize the ClassifierCriterion.

		:param inverse: Whether the criterion should be inverted.
		:param allow_batched: Whether the criterion supports batching.
		"""

		super().__init__(inverse, allow_batched)

		# Wrap the evaluate method with logging (replaces criteria_kwargs decorator)
		if not self._allow_batched:
			original_evaluate = self.evaluate

			def logged_evaluate(
					_: Any, *, handles: Tensor, targets: Tensor, img_resolution: int, **kwargs: Any
			) -> Union[float, list[float]]:
				return original_evaluate(handles=handles, targets=targets, img_resolution=img_resolution, **kwargs)

			self.evaluate = logged_evaluate.__get__(self, self.__class__)  # type: ignore[method-assign]

		if self._allow_batched:
			eval_func = self.evaluate

			def batched_evaluate(
					_: Any,
					*,
					handles: Tensor,
					targets: Tensor,
					img_resolution: int,
					batch_dim: Union[int, None] = None,
					**kwargs: Any,
			) -> Union[float, list[float]]:
				if batch_dim is None:
					handles = handles.unsqueeze(0)
					targets = targets.unsqueeze(0)
				elif batch_dim != 0:
					handles = handles.transpose(0, batch_dim)
					targets = targets.transpose(0, batch_dim)

				results = eval_func(handles=handles, targets=targets, img_resolution=img_resolution, **kwargs)

				return results[0] if batch_dim is None and isinstance(results, list) else results

			self.evaluate = batched_evaluate.__get__(self, self.__class__)  # type: ignore[method-assign]

	@abstractmethod
	def evaluate(
			self, *, handles: Tensor, targets: Tensor, img_resolution: int, **kwargs: Any
	) -> Union[float, list[float]]:
		"""
		Evaluate the criterion in question.

		:param handles: The handle coordinates of the operations.
		:param targets: The target coordinates of the operations.
		:param img_resolution: Image resolution of the generator to normalize the distance.
		:param kwargs: Other keyword arguments passed to the criterion.
		:returns: The value(s).
		"""

		...