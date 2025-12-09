from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .._candidate import Candidate, CandidateList


@dataclass
class DragCandidate(Candidate):
    """A simple container for candidate elements used in dragging operations."""

    handles: list[tuple[int, int]] # (Y, X) coordinates of the handle points.
    targets: list[tuple[int, int]] # (Y, X) coordinates of the target points.


class DragCandidateList(CandidateList):
    """
    A custom list like object to handle DragCandidates easily.

    Note: This list object is immutable and caches getters.
    """

    _handles: Optional[list[list[tuple[int, int]]]]
    _targets: Optional[list[list[tuple[int, int]]]]

    def __init__(self, *initial_candidates: DragCandidate) -> None:
        super().__init__(*initial_candidates)

        self._handles = [elem.handles for elem in self.data]
        self._targets = [elem.targets for elem in self.data]

    @property
    def handles(self) -> list[list[tuple[int, int]]]:
        return self._handles

    @property
    def targets(self) -> list[list[tuple[int, int]]]:
        return self._targets
