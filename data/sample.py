# sampler.py

import numpy as np
import cocoex  # COCO experiment module from NumBBO

# Load the BBOB suite (Suite 1: single-objective noiseless benchmarks)
suite = cocoex.Suite("bbob", "", "")  # empty strings mean default instances/dimensions
# We can filter or select specific problem by index if needed:
# e.g., get the Sphere function (fid=1) in 5-D, instance 1
def load_bbob_function(func_id: int, dimension: int, instance: int = 1):
    problem = suite.get_problem_by_function_dimension_instance(func_id, dimension, instance)
    # `problem` can be called like problem(x) to get a function value.
    info = f"Function {func_id} (dim={dimension}, instance={instance}): {problem.name}"
    return problem, info

def initial_sample(problem, n_samples: int, lower_bound: float, upper_bound: float):
    dim = problem.dimension  # dimension of the problem
    # Sample uniformly within [lower_bound, upper_bound] for each dimension
    X_init = np.random.uniform(lower_bound, upper_bound, size=(n_samples, dim))
    # Evaluate the expensive function on these initial points
    y_init = np.array([problem(x) for x in X_init], dtype=float)
    return X_init, y_init
