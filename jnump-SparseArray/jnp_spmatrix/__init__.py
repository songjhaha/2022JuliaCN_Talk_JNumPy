from jnumpy import init_jl, init_project

def enable_jnp_spmatrix():
    init_jl()
    init_project(__file__)
    return  

