module Quick_Kappa

using KrylovKit
import LinearAlgebra: adjoint

# -----------------------------------------------------------------------------
#  Background functions
# -----------------------------------------------------------------------------

"""
    VirtualInverse(mat::AbstractMatrix)

Wraps a matrix to present its inverse (and adjoint-inverse) as a
matrix-free linear operator, for use with KrylovKit.
"""
struct VirtualInverse{T<:AbstractMatrix}
    mat::T
end

# Forward solve:  y = mat \ x
function (V::VirtualInverse)(x::AbstractVecOrMat)
    return V.mat \ x
end

# Adjoint solve: y = matᵀ \ x
function (V::VirtualInverse)(x::AbstractVecOrMat, ::Val{true})
    return adjoint(V.mat) \ x
end

# Explicitly handle `Val{false}` too, for symmetry
function (V::VirtualInverse)(x::AbstractVecOrMat, ::Val{false})
    return V.mat \ x
end

"""
    adjoint(V::VirtualInverse) -> VirtualInverse

The adjoint of a VirtualInverse just wraps the adjoint of the underlying matrix.
"""
function adjoint(V::VirtualInverse)
    return VirtualInverse(adjoint(V.mat))
end

"""
    VirtualPrecon(P::AbstractMatrix, A::AbstractMatrix)

Wraps the preconditioned operator x ↦ P⁻¹ * (A * x) and its adjoint
for matrix-free SVD of P⁻¹A.
"""
struct VirtualPrecon{Pmat<:AbstractMatrix, Amat<:AbstractMatrix}
    P::Pmat
    A::Amat
end

"""
    (V::VirtualPrecon)(x, ::Val{false})

Apply the forward preconditioned operator:  P⁻¹ (A * x).
"""
function (V::VirtualPrecon)(x::AbstractVecOrMat, ::Val{false})
    return V.P \ (V.A * x)
end

"""
    (V::VirtualPrecon)(x, ::Val{true})

Apply the adjoint preconditioned operator:  Aᵀ (P⁻ᵀ * x).
"""
function (V::VirtualPrecon)(x::AbstractVecOrMat, ::Val{true})
    return adjoint(V.A) * (adjoint(V.P) \ x)
end

# -----------------------------------------------------------------------------
#  Singular-value and condition-number functions
# -----------------------------------------------------------------------------

"""
    maxsv(A::AbstractMatrix) -> σ_max(A)

Approximate the largest singular value of `A` using a rank-1 Krylov SVD.
"""
function maxsv(A::AbstractMatrix)
    sv, _ = svdsolve(A, 1)
    return sv[1]
end

"""
    maxsv(P::AbstractMatrix, A::AbstractMatrix) -> σ_max(P⁻¹A)

Approximate the largest singular value of the preconditioned matrix P⁻¹A.
"""
function maxsv(P::AbstractMatrix, A::AbstractMatrix)
    sv, _ = svdsolve(VirtualPrecon(P, A), size(A,1), 1)
    return sv[1]
end

"""
    minsv(A::AbstractMatrix) -> σ_min(A)

Approximate the smallest singular value of `A` by applying SVD to the
inverse operator.
"""
function minsv(A::AbstractMatrix)
    sv, _ = svdsolve(VirtualInverse(A), size(A,1), 1)
    return 1 / sv[1]
end

"""
    minsv(P::AbstractMatrix, A::AbstractMatrix) -> σ_min(P⁻¹A)

Approximate the smallest singular value of the preconditioned matrix P⁻¹A.
"""
function minsv(P::AbstractMatrix, A::AbstractMatrix)
    # we invert the roles to get the smallest of P⁻¹A
    svinv, _ = svdsolve(VirtualPrecon(A, P), size(A,1), 1)
    return 1 / svinv[1]
end

"""
    kcond(A::AbstractMatrix) -> κ(A)

Approximate the condition number of `A` via σ_max(A) · σ_max(A⁻¹).
"""
function kcond(A::AbstractMatrix)
    σmax, _ = svdsolve(A, 1)
    σinv, _ = svdsolve(VirtualInverse(A), size(A,1), 1)
    return σmax[1] * σinv[1]
end

"""
    kprecond(P::AbstractMatrix, A::AbstractMatrix) -> κ(P⁻¹A)

Approximate the condition number of the preconditioned operator P⁻¹A.
"""
function kprecond(P::AbstractMatrix, A::AbstractMatrix)
    σmax, _   = svdsolve(VirtualPrecon(P, A), size(A,1), 1)
    σinv, _   = svdsolve(VirtualPrecon(A, P), size(A,1), 1)
    return σmax[1] * σinv[1]
end



# -----------------------------------------------------------------------------
#  Tests
# -----------------------------------------------------------------------------
# using Test, LinearAlgebra

# # 1) Identity ⇒ κ = 1
# @test isapprox(kcond(Matrix{Float64}(I, 3, 3)), 1.0; rtol=1e-8)

# # 2) Simple SPD ⇒ match Base.cond
# A2 = [3.0 1.0; 1.0 2.0]
# @test isapprox(kcond(A2), cond(A2); rtol=1e-6)

# # 3) Random ⇒ match full SVD
# M5 = randn(5,5)
# S   = svd(M5).S
# @test isapprox(maxsv(M5), S[1];   rtol=1e-6)
# @test isapprox(minsv(M5), S[end]; rtol=1e-6)

# # 4) Preconditioned vs direct
# P2     = Diagonal([2.0, 3.0])
# direct = svd(P2 \ A2).S
# @test isapprox(maxsv(P2, A2), direct[1]; rtol=1e-6)
# @test isapprox(minsv(P2, A2), direct[end]; rtol=1e-6)

# println("All tests passed.")


export maxsv, minsv, kcond, kprecond
end