import Random
using Random: MersenneTwister
using Distributions
using Test
using StaticArrays
using LinearAlgebra
using Unitful


μ = rand()
x = rand()
σ = rand()
Σ = σ*σ'

# Check type conversions
GFloat = Gaussian{Vector{Float64}, Matrix{Float64}}
v = GFloat[Gaussian([1.0], Matrix(1.0I, 2, 2)),
           Gaussian(SVector(1.0), @SMatrix [1.0 0.0; 0.0 1.0])]
@test mean.(v) == [[1.0], [1.0]]

G = Gaussian([1.0], Matrix(1.0I, 2, 2))
m, K = G
@test m === mean(G)
@test K === cov(G)

p = pdf(Normal(μ, √Σ), x)
@test pdf(Gaussian(μ, Σ), x) ≈ p
@test pdf(Gaussian(μ, Σ*I), x) ≈ p
@test pdf(Gaussian([μ], [σ]*[σ]'), x) ≈ p

@test pdf(Gaussian((@SVector [μ]), @SMatrix [Σ]), @SVector [x]) ≈ p

for d in 1: 3
    local μ = rand(d)
    local x = rand(d)
    local σ = tril(rand(d,d))
    local Σ = σ*σ'
    local p = pdf(MvNormal(μ, Σ), x)

    @test pdf(Gaussian(μ, Σ), x) ≈ p
    @test pdf(Gaussian(μ, PSD(σ)), x) ≈ p
    @test pdf(Gaussian(SVector{d}(μ), SMatrix{d,d}(Σ)), x) ≈ p
    @test pdf(Gaussian(SVector{d}(μ), PSD(SMatrix{d,d}(σ))), x) ≈ p
end

for d in 1: 3
    local μ = rand(d)
    local x = rand(d)
    local σ = rand()
    local Σ = Matrix(1.0I, d, d).*σ^2
    local p = pdf(MvNormal(μ, Σ), x)

    @test pdf(Gaussian(μ, σ^2*I), x) ≈ p
    @test pdf(Gaussian(SVector{d}(μ), SDiagonal(σ^2*ones(SVector{d}))), x) ≈ p
    @test pdf(Gaussian(SVector{d}(μ), SMatrix{d,d}(Σ)), x) ≈ p
end

@test rand(Random.GLOBAL_RNG, Gaussian(1.0, 0.0)) == 1.0
@test mean(rand(MersenneTwister(1), Gaussian([1., 2], Matrix(1.0I, 2, 2)), 100000)) ≈ 1.5 atol=0.02
@test mean(rand(MersenneTwister(1), Gaussian(1.5u"m", 2.0u"m^2"), 100000)) ≈ 1.5u"m" atol=0.02u"m"

@test rand(Gaussian(1.0,0.0)) == 1.0
Random.seed!(5)
x = randn()
Random.seed!(5)
@test rand(Gaussian(1.0,2.0)) == 1.0 .+ sqrt(2)*x

Random.seed!(5)
@test rand(Gaussian(1.0,2.0), (1,)) == [1.0 + sqrt(2)*x]

g = Gaussian([1,2], Matrix(1.0I, 2, 2))
@test mean(g + 10) == [11,12]
@test g - 10 + 10 == g
@test mean(g + [10, 20]) == [11,22]
m = [1.0 2.0; 0.0 2.0]
@test cov(m * g) == m * m'

using GaussianDistributions: ⊕
@test cov(m*g ⊕ g) == cov(m*g) + cov(g)
@test mean(m*g ⊕ g) == mean(m*g) + mean(g)

@test cov(g ⊕ m*mean(g)) == cov(g)
@test mean(m*mean(g) ⊕ g) == mean(m*g) + mean(g)

g2 = Gaussian(SVector(1.000001, 2.0), [1.00001 0.0; 0.0 0.999999])
@test !(g ≈ g2)
@test g ≈ g2 rtol=0.0001
