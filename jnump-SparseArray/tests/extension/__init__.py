from jnp_spmatrix import enable_jnp_spmatrix
from jnumpy import init_jl, init_project

init_jl()
init_project(__file__)
enable_jnp_spmatrix()

from _extension import jl_spmatrix_to_matrix, jl_matrix_to_spmatrix # type: ignore


