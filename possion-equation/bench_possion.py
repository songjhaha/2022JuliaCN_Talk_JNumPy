# running in ipython
#%%
import numpy as np
import scipy
from scipy.sparse import diags
from scipy.sparse.linalg import gmres, LinearOperator
from numba import jit as njit
from possion import jl_solve_possion

#%%
# solve possion equation
# Δu = f(x)
# with finite difference method
# we will solve a linear system
# Au = b
x = np.linspace(0, 1, 1000)[1:-1]
rhs = np.sin(2 * np.pi * x)
dx = x[1] - x[0]

#%%
# sparse matrix
def scipy_solve_possion(rhs: np.ndarray, dx: float, reltol: float = 1e-5, maxiter: int = 1000) -> np.ndarray:
    A = diags([1, -2, 1], [-1, 0, 1], shape=(rhs.size, rhs.size), format="csr") / dx**2
    x0 = np.zeros_like(rhs)
    return gmres(A, rhs, x0=x0, tol=reltol, maxiter=maxiter)[0]

# 558 µs
%timeit -n10000 scipy_solve_possion(rhs, dx, reltol=1e-5, maxiter=1000)

#%%
# matrix free
def build_A(x):
    Nx = x.shape[0]
    dx = x[1] - x[0]
    @njit
    def mv(v):
        out = np.zeros(Nx)
        out[0] = (- 2 * v[0] + v[1]) / dx**2
        out[-1] = (v[-2] - 2 * v[-1]) / dx**2
        for i in range(1, Nx - 1):
            out[i] = (v[i - 1] - 2 * v[i] + v[i + 1]) / dx**2
        return out

    A = LinearOperator((Nx, Nx), matvec=mv)
    return A

A = build_A(x)

def scipy_solve_possion2(A: LinearOperator, rhs: np.ndarray, dx: float, reltol: float = 1e-5, maxiter: int = 1000) -> np.ndarray:
    x0 = np.zeros_like(rhs)
    return gmres(A, rhs, x0=x0, tol=reltol, maxiter=maxiter)[0]

# 135 µs
%timeit -n10000 scipy_solve_possion2(A, rhs, dx, reltol=1e-5, maxiter=1000)

# %%
# pycall
from julia import Main as jl
jl.include(r'possion/src/possion.jl')
_pycall_solve_possion = jl.possion.solve_possion

def pycall_solve_possion(rhs: np.ndarray, dx: float, reltol: float = 1e-5, maxiter: int = 1000) -> np.ndarray:
    x0 = np.zeros_like(rhs)
    return _pycall_solve_possion(x0, rhs, dx, reltol, maxiter)

def time_pycall_solve_possion():
    pycall_solve_possion(rhs, dx, reltol=1e-5, maxiter=1000)
# 61.3 µs
%timeit -n10000 pycall_solve_possion(rhs, dx, reltol=1e-5, maxiter=1000)

# %%
# jnumpy
# 50.3 µs
%timeit -n10000 jl_solve_possion(rhs, dx, reltol=1e-5, maxiter=1000)

# %%
# check residual
sol_jl = jl_solve_possion(rhs, dx, reltol=1e-5, maxiter=1000)
sol_scipy = scipy_solve_possion(rhs, dx, reltol=1e-5, maxiter=1000)
scipy.linalg.norm(sol_jl - sol_scipy)
scipy.linalg.norm(A @ sol_jl - rhs)
scipy.linalg.norm(A @ sol_scipy - rhs)



