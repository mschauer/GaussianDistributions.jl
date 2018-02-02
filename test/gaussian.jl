using Distributions
using Base.Test
using StaticArrays


μ = rand()
x = rand()
σ = rand()
Σ = σ*σ'

p = pdf(Normal(μ, √Σ), x)
@test pdf(Gaussian(μ, Σ), x) ≈ p
@test pdf(Gaussian(μ, Σ*I), x) ≈ p
@test pdf(Gaussian([μ], [σ]*[σ]'), x) ≈ p

@test pdf(Gaussian((@SVector [μ]), @SMatrix [Σ]), @SVector [x]) ≈ p

for d in 1: 3
    μ = rand(d)
    x = rand(d)
    σ = tril(rand(d,d))
    Σ = σ*σ'
    p = pdf(MvNormal(μ, Σ), x)

    @test pdf(Gaussian(μ, Σ), x) ≈ p
    @test pdf(Gaussian(μ, PSD(σ)), x) ≈ p
    @test pdf(Gaussian(SVector{d}(μ), SMatrix{d,d}(Σ)), x) ≈ p
    @test pdf(Gaussian(SVector{d}(μ), PSD(SMatrix{d,d}(σ))), x) ≈ p
end

for d in 1: 3
    μ = rand(d)
    x = rand(d)
    σ = rand()
    Σ = eye(d)*σ^2
    p = pdf(MvNormal(μ, Σ), x)

    @test pdf(Gaussian(μ, σ^2*I), x) ≈ p
    @test pdf(Gaussian(SVector{d}(μ), SDiagonal(σ^2*ones(SVector{d}))), x) ≈ p
    @test pdf(Gaussian(SVector{d}(μ), SMatrix{d,d}(Σ)), x) ≈ p
end

@test rand(Base.Random.GLOBAL_RNG, Gaussian(1.0, 0.0)) == 1.0

@test rand(Gaussian(1.0,0.0)) == 1.0
srand(5)
x = randn()
srand(5)
@test rand(Gaussian(1.0,2.0)) == 1.0 + sqrt(2)*x

srand(5)
@test rand(Gaussian(1.0,2.0), (1,)) == [1.0 + sqrt(2)*x]
