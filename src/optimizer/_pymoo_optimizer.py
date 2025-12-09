import logging
from typing import Any, Optional, Type

import numpy as np
from numpy.typing import NDArray
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem

from ._optimizer import Optimizer
from .auxiliary_components import OptimizerCandidate


class PymooOptimizer(Optimizer):
	"""A Learner class for easy Pymoo integration"""

	_pymoo_algo: GeneticAlgorithm
	_problem: Problem
	_pop_current: Population
	_bounds: tuple[int, int]
	_shape: tuple[int, ...]

	_params: dict[str, Any]
	_algorithm: Type[GeneticAlgorithm]

	def __init__(
			self,
			bounds: tuple[int, int],
			algorithm: Type[GeneticAlgorithm],
			algo_params: dict[str, Any],
			num_objectives: int,
			num_drags: int,
			target_coord_system
	) -> None:
		"""
		Initialize the genetic learner.

		:param bounds: Bounds for the optimizer.
		:param algorithm: The pymoo Algorithm.
		:param algo_params: Parameters for the pymoo Algorithm.
		:param num_objectives: The number of objectives the learner can handle.
		:param num_drags: The number of dragging operations for solutions.
		:param target_coord_system: Whether the target coordinates are in Cartesian or Polar coordinates.
		"""

		super().__init__(num_objectives)
		# Initialize Constants.
		self._params = algo_params
		self._algorithm = algorithm
		self._bounds = bounds
		self._num_drags = num_drags
		self._target_coord_system = target_coord_system

		# Initialize optimization problem and initial solutions.
		self.update_problem(self._target_coord_system)
		self._optimizer_type = type(self._pymoo_algo)

	def update(self) -> None:
		"""
		Generate a new population.
		"""
		logging.info("Sampling new population...")
		static = StaticProblem(self._problem, F=np.column_stack(self._fitness))
		Evaluator().eval(static, self._pop_current)
		self._pymoo_algo.tell(self._pop_current)

		self._pop_current = self._pymoo_algo.ask()
		self._x_current = self._clip_to_bounds(self._pop_current.get("X"))

	def get_x_current(self) -> NDArray:
		"""
		Return the current population in a specific format.

		:return: The current best genome.
		"""

		return self._x_current.reshape((self._x_current.shape[0], *self._shape))

	def update_problem(self, target_coord_system: str = None) -> None:
		"""
		Change problem shape of optimization.

		:param target_coord_system: Whether the target coordinates are in Cartesian or Polar coordinates.
		"""

		target_coord_system = target_coord_system or self._target_coord_system
		self._target_coord_system = target_coord_system

		# Each dragging operation has a handle and a target with X and Y pairs.
		solution_shape = (self._num_drags, 2, 2)

		sampling = self.random_sample(self._params.get("pop_size"), solution_shape, target_coord_system)
		sampling = sampling.reshape(sampling.shape[0], -1)
		sampling = self._clip_to_bounds(sampling)
		self._params["sampling"] = sampling

		self._shape = solution_shape
		self._n_var = int(np.prod(solution_shape))
		self._pymoo_algo = self._algorithm(**self._params, save_history=True)

		self._problem = Problem(n_var=self._n_var,
								n_obj=self._num_objectives,
								xl=self._bounds[0],
								xu=self._bounds[1],
								vtype=float)
		self._pymoo_algo.setup(self._problem, termination=NoTermination())

		self.reset()

	def reset(self) -> None:
		"""Resets the optimizer."""
		self._pop_current = self._pymoo_algo.ask()
		self._x_current = self._clip_to_bounds(self._pop_current.get("X"))

		self._best_candidates = [
			OptimizerCandidate(
				solution=np.random.uniform(low=self._bounds[0], high=self._bounds[1], size=self._n_var),
				fitness=[np.inf] * self._num_objectives,
			)
		]
		self._previous_best = self._best_candidates.copy()

	def random_sample(self, n_samples: int, solution_shape: tuple[int, ...], target_coord_system: str = "cartesian") -> NDArray:
		sampling = np.zeros((n_samples, *solution_shape))

		for i in range(n_samples):
			sol = np.zeros(solution_shape)

			sol[:, 0, :] = np.random.uniform(low=self._bounds[0], high=self._bounds[1], size=(self._num_drags, 2))
			sol[:, 1, 0] = np.random.uniform(low=self._bounds[0], high=self._bounds[1], size=self._num_drags)

			if target_coord_system == "cartesian":
				sol[:, 1, 1] = np.random.uniform(low=self._bounds[0], high=self._bounds[1], size=self._num_drags)
			elif target_coord_system == "polar":
				sol[:, 1, 1] = np.abs(np.random.normal(scale=0.4, size=self._num_drags))

			sampling[i] = sol

		return sampling

	@property
	def best_solutions_reshaped(self) -> NDArray:
		"""
		Get the best solutions in correct shape.

		:return: The solutions.
		"""

		return np.asarray([
			c.solution.reshape(self._shape) for c in self._best_candidates if c.solution is not None
		])
