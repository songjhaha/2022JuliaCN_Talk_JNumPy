module possion
using TyPython
using TyPython.CPython
import LinearAlgebra
import IterativeSolvers

struct Laplace
    dx::Float64
    size::Tuple{Int, Int}
end

Base.size(A::Laplace) = A.size
Base.size(A::Laplace, i::Int) = size(A)[i]
Base.eltype(A::Laplace) = Float64

function LinearAlgebra.mul!(C, A::Laplace, B)
    @inbounds begin
        for i in 2:length(B)-1
            C[i] = (B[i-1] - 2B[i] + B[i+1])/A.dx^2
        end
        C[1] = (-2B[1] + B[2])/A.dx^2
        C[end] = (B[end-1] - 2B[end])/A.dx^2
    end
    return C
end
Base.:*(A::Laplace, B::AbstractVector) = (C = similar(B); LinearAlgebra.mul!(C,A,B))

@export_py function solve_possion(x::Vector{Float64}, b::Vector{Float64}, dx::Float64, reltol::Float64, maxiter::Int)::Vector{Float64}
    A = Laplace(dx, (length(b), length(b)))
    IterativeSolvers.gmres!(x, A, b; reltol = reltol, maxiter = maxiter, restart = 20)
    res = A*x - b
    return x
end

function init()
    @export_pymodule _possion begin
        _jl_solve_possion = Pyfunc(solve_possion)
    end
end
precompile(init, ())
end
