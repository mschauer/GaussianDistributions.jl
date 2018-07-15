__precompile__()
module GaussianDistributions

# Gaussian
using Distributions
using Base.LinAlg: norm_sqr

import Base: rand, mean, cov, var
import Distributions: pdf, logpdf, sqmahal, cdf, quantile
import Base: chol, size

export PSD, Gaussian

"""
    PSD{T}

Simple wrapper for the lower triangular Cholesky root of a positive (semi-)definite element `σ`.
"""
mutable struct PSD{T}
    σ::T
    PSD(σ::T) where {T} = istril(σ) ? new{T}(σ) : throw(ArgumentError("Argument not lower triangular"))
end
chol(P::PSD) = P.σ' 

"""
Sum of the log of the diagonal elements. Second argument `d` is used
to handle `UniformScaling` and other linear operators whose dimensions
are determined by the dimension of the argument they work on.
"""
sumlogdiag(Σ::Float64, _) = log(Σ)
sumlogdiag(Σ, d) = sum(log.(diag(Σ)))
sumlogdiag(J::UniformScaling, d) = log(J.λ)*d

_logdet(Σ::PSD, d) = 2*sumlogdiag(Σ.σ, d)

_logdet(Σ, d) = logdet(Σ)
_logdet(J::UniformScaling, d) = log(J.λ) * d

_symmetric(Σ) = Symmetric(Σ)
_symmetric(J::UniformScaling) = J

"""
    Gaussian(μ, Σ) -> P

Gaussian distribution with mean `μ` and covariance `Σ`. Defines `rand(P)` and `(log-)pdf(P, x)`.
Designed to work with `Number`s, `UniformScaling`s, `StaticArrays` and `PSD`-matrices.

Implementation details: On `Σ` the functions `logdet`, `whiten` and `unwhiten`
(or `chol` as fallback for the latter two) are called.
"""
struct Gaussian{T,S}
    μ::T
    Σ::S
    Gaussian(μ::T, Σ::S) where {T,S} = new{T,S}(μ, Σ)
end
Base.:(==)(g1::Gaussian, g2::Gaussian) = g1.μ == g2.μ && g1.Σ == g2.Σ
Gaussian() = Gaussian(0.0, 1.0)
mean(P::Gaussian) = P.μ
cov(P::Gaussian) = P.Σ
var(P::Gaussian{<:Real}) = P.Σ
Base.convert(::Type{Gaussian{T, S}}, g::Gaussian) where {T, S} =
    Gaussian(convert(T, g.μ), convert(S, g.Σ))
     
dim(P::Gaussian) = length(P.μ)
whiten(Σ::PSD, z) = Σ.σ\z
whiten(Σ, z) = chol(Σ)'\z
whiten(Σ::UniformScaling, z) = z/sqrt(Σ.λ)
sqmahal(P::Gaussian, x) = norm_sqr(whiten(P.Σ, x - P.μ))

rand(P::Gaussian) = P.μ + chol(P.Σ)'*randn(typeof(P.μ))
rand(P::Gaussian{Vector{T}}) where T =
    P.μ + chol(P.Σ)'*randn(T, length(P.μ))
rand(RNG, P::Gaussian) = P.μ + chol(P.Σ)'*randn(RNG, typeof(P.μ))
rand(RNG, P::Gaussian{Vector{T}}) where T =
    P.μ + chol(P.Σ)'*randn(RNG, T, length(P.μ))

logpdf(P::Gaussian, x) = -(sqmahal(P,x) + _logdet(P.Σ, dim(P)) + dim(P)*log(2pi))/2    
pdf(P::Gaussian, x) = exp(logpdf(P::Gaussian, x))
cdf(P::Gaussian{Number}, x) = Distributions.normcdf(P.μ, sqrt(P.Σ), x)

Base.:+(g::Gaussian, vec) = Gaussian(g.μ + vec, g.Σ)
Base.:+(vec, g::Gaussian) = g + vec
Base.:-(g::Gaussian, vec) = g + (-vec)
Base.:*(M, g::Gaussian) = Gaussian(M * g.μ, M * g.Σ * M')

function rand(RNG, P::Gaussian{T}, dims) where {T}
    X = zeros(T, dims)
    for i in 1:length(X)
        X[i] = rand(RNG, P)
    end
    X
end

function rand(RNG, P::Gaussian{Vector{T}}, dims) where {T}
    X = zeros(T, dim(P), dims...)
    for i in 1:prod(dims)
        X[:, i] = rand(RNG, P)
    end
    X
end

rand(P::Gaussian, dims) = rand(Base.GLOBAL_RNG, P, dims)

"""
    logpdfnormal(x, Σ) 

Logarithm of the probability density function of centered Gaussian with covariance Σ
"""
function logpdfnormal(x, Σ) 

    S = chol(_symmetric(Σ))'

    d = length(x)
     -((norm(S\x))^2 + 2sumlogdiag(S,d) + d*log(2pi))/2
end
function logpdfnormal(x::Float64, Σ) 
     -(x^2/Σ + log(Σ) + log(2pi))/2
end

"""
    conditional(P::Gaussian, A, B, xB)  

Conditional distribution of `X[i for i in A]` given 
`X[i for i in B] == xB` if ``X ~ P``.
"""
function conditional(P::Gaussian, A, B, xB)
    Z = P.Σ[A,B]*inv(P.Σ[B,B]) 
    Gaussian(P.μ[A] + Z*(xB - P.μ[B]), P.Σ[A,A] - Z*P.Σ[B,A])
end    

include("bivariate.jl")

end # module
