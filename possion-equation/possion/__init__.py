import numpy as np
from jnumpy import init_jl, init_project
from scipy.sparse import diags
from scipy.sparse.linalg import gmres, LinearOperator
from numba import jit as njit

init_jl()
init_project(__file__)

from _possion import _jl_solve_possion # type: ignore


def jl_solve_possion(rhs: np.ndarray, dx: float, reltol: float = 1e-5, maxiter: int = 1000) -> np.ndarray:
    x0 = np.zeros_like(rhs)
    return _jl_solve_possion(x0, rhs, dx, reltol, maxiter)



