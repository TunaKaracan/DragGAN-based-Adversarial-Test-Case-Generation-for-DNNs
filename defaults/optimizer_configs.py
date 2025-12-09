"""A collection of default parameters for the learner classes."""

from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.operators.sampling.lhs import LHS

PYMOO_AGE_MOEA_DEFAULT_PARAMS = {
    "bounds": (0, 1),
    "algorithm": AGEMOEA2,
    "algo_params": {
        "sampling": LHS(),
        "pop_size": 50,  # 100 is the default in pymoo.
    },
    "num_drags": 4
}
