from scipy import sparse
import numpy as np
from .extension import jl_spmatrix_to_matrix, jl_matrix_to_spmatrix

def test_spmatrix_to_matrix():
    A = sparse.random(10, 10, density=0.5)
    B = jl_spmatrix_to_matrix(A)
    assert isinstance(B, np.ndarray)
    assert np.array_equal(A.toarray(), B)
    return

def test_matrix_to_spmatrix():
    A = sparse.random(10, 10, density=0.5)
    B = jl_matrix_to_spmatrix(A.toarray())
    assert sparse.isspmatrix_csc(B)
    assert np.array_equal(A.toarray(), B.toarray())
    return