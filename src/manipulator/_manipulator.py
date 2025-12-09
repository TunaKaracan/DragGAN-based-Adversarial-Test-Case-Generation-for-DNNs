from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor

from ._candidate import Candidate, CandidateList


class Manipulator(ABC):
    """An abstract manipulator class."""

    @abstractmethod
    def manipulate(self, w0: Tensor, candidates: CandidateList[Candidate], **kwargs: Any) -> Any:
        """
        The manipulation function for the Manipulator.

        :param w0: The w of the initial image to manipulate.
        :param candidates: The candidates to manipulate.
        :param kwargs: Keyword arguments to pass to the manipulation function.
        :returns: The result of the manipulation.
        """
        ...