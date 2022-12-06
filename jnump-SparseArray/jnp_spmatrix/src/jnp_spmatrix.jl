module jnp_spmatrix

using TyPython
using TyPython.CPython
import TyPython.CPython: py_cast, py_coerce, py_import, Py, UnsafeNew, G_PyBuiltin, PyAPI, unsafe_set!, py_seterror!, py_throw
using SparseArrays

const scipysparse = Py(UnsafeNew())

function CPython.py_coerce(::Type{SparseMatrixCSC}, o::Py)
    if !py_coerce(Bool, scipysparse.issparse(o))
        py_seterror!(G_PyBuiltin.TypeError, "expected sparse matrix type")
        py_throw()
    end
    row, col, data = py_coerce(Tuple{Vector, Vector, Vector}, scipysparse.find(o))
    I = row .+ 1
    J = col .+ 1
    return sparse(I, J, data)
end

function CPython.py_cast(::Type{Py}, o::SparseMatrixCSC)
    rowval, colptr, val = o.rowval, o.colptr, o.nzval
    py_indices = py_cast(Py, rowval .- 1)
    py_indptr = py_cast(Py, colptr .- 1)
    py_data = py_cast(Py, val)
    py_shape = py_cast(Py, size(o))
    return scipysparse.csc_matrix(py_cast(Py, (py_data, py_indices, py_indptr)), shape=py_shape)
end

function init()
    scipysp = PyAPI.PyImport_ImportModule("scipy.sparse")
    unsafe_set!(scipysparse, scipysp)
end

precompile(init, ())

end
