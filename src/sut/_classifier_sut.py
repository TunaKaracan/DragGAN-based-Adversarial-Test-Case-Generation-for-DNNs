from typing import Optional

import torch
from torch import Tensor, nn

from ._sut import SUT
from .auxiliary_components import MonteCarloDropoutScaffold


class ClassifierSUT(SUT):
    """A classifier SUT."""

    _model: nn.Module
    _softmax: nn.Softmax
    _sigmoid: nn.Sigmoid

    _apply_activation: str
    _batch_size: int

    def __init__(
        self,
        model: nn.Module,
        transformer: nn.Module = None,
        apply_activation: str = "none",
        use_mcd: bool = False,
        batch_size: int = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize the classifier SUT.

        :param model: The model to use.
        :param transformer: The transformer to apply to the inputs.
        :param apply_activation: Which activation to apply to the logits. "none", "softmax" or "sigmoid".
        :param use_mcd: Whether to use Monte Carlo Dropout or not.
        :param batch_size: The batch size to use for prediction.
        :param device: The device to use if available.
        """

        self._apply_activation = apply_activation
        self._batch_size = batch_size
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = MonteCarloDropoutScaffold(model) if use_mcd else model
        self._model.eval()
        self._transformer = transformer or nn.Identity()
        self._softmax = nn.Softmax(dim=-1)
        self._sigmoid = nn.Sigmoid()

        self._model.to(self._device)
        self._transformer.to(self._device)
        self._softmax.to(self._device)
        self._sigmoid.to(self._device)

    @torch.no_grad()
    def process_input(self, inpt: Tensor) -> Tensor:
        """
        Predict class probabilities from input.

        :param inpt: Input tensor.
        :return: Predicted class probabilities on CPU.
        """

        if inpt.device != self._device:
            inpt = inpt.to(self._device)

        batch_size = max(
            self._batch_size or inpt.size(0), 1
        )  # If batchsize == 0 -> do whole input.
        n_chunks = (inpt.size(0) + batch_size - 1) // batch_size
        chunks = torch.chunk(inpt, n_chunks)

        assert torch.isfinite(inpt).all(), "input has NaNs/Infs"

        results = []
        for c in chunks:
            logits = self._model(self._transformer(c))

            if self._apply_activation == "softmax":
                output = self._softmax(logits)
            elif self._apply_activation == "sigmoid":
                output = self._sigmoid(logits)
            elif self._apply_activation == "none":
                output = logits
            else:
                raise ValueError(f"Unknown activation function {self._apply_activation}!")

            results.append(output.detach().cpu())

        return torch.cat(results)
