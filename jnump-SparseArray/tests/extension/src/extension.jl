module extension

using TyPython
using TyPython.CPython
using SparseArrays

@export_py function spmatrix_to_matrix(a::SparseMatrixCSC)::Array
    return collect(a)
end

@export_py function matrix_to_spmatrix(a::AbstractMatrix)::SparseMatrixCSC
    return sparse(a)
end

function init()
    @export_pymodule _extension begin
        jl_spmatrix_to_matrix = Pyfunc(spmatrix_to_matrix)
        jl_matrix_to_spmatrix = Pyfunc(matrix_to_spmatrix)
    end
end

precompile(init, ())

end
