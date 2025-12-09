from typing import Any

import torch
from sympy.codegen.ast import Raise
from torch import Tensor

from ._image_criterion import ImageCriterion


class MatrixDistance(ImageCriterion):
    """Implements a channel-wise matrix distance measure based on torch.linalg.norm."""

    _name: str = "MatD"
    _all_norms: list[str] = ["fro", "nuc", "inf", "-inf", "1", "-1", "2", "-2"]

    def __init__(self, inverse: bool = False, norm: str = "fro") -> None:
        """
        Initialize the MatrixDistance criterion.

        :param inverse: Whether the measure should be inverted (default: False).
        :param norm: Which norm to use (default: fro).
        """

        super().__init__(inverse, allow_batched=True)
        assert norm in self._all_norms, f"Norm {norm} not in supported norms: {self._all_norms}"
        self.norm = norm
        self._name += f"_{norm}"

    @torch.no_grad()
    def evaluate(self, *, images: Tensor, **_: Any) -> list[float]:
        """
        Calculate the normalized matrix distance between two tensors that are in range [0, 1].

        :param images: Images to compare, first image is the original image the rest will be compared to.
        :param _: Additional unused kwargs.
        :returns: The distance per channel.
        """

        orig_img, other_imgs = images[:1], images[1:]

        # Expect the image tensors to have shape: B x C x H x W
        # Upper bound of distance.
        ub = torch.linalg.matrix_norm(torch.ones_like(orig_img), self.norm, dim=(-2, -1))

        diffs = orig_img - other_imgs
        norm = torch.linalg.matrix_norm(diffs, self.norm, dim=(-2, -1))
        scaled = norm / ub

        channel_wise = scaled.mean(dim=1)
        results: list[float] = torch.abs(self._inverse.real - channel_wise).float().tolist()
        return results
