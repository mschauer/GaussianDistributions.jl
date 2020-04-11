using GaussianDistributions
using Random
using Test
using Statistics
using LinearAlgebra

n = 500
Random.seed!(2)
L1 = rand(3,3)
Σ = L1*L1'
L2 = rand(2,2)
R = L2*L2'
H = rand(2,3)
μ = rand(3)
X = L1*randn(3, n)
Y = H*X + L2*randn(2, n)

@test norm(cov(vcat(X,Y)') - [L1*L1' L1*L1'*H' ; H*L1*L1' (H*L1*L1'*H' + L2*L2')]) < 20/sqrt(n)

function correct2(x, (y, R), H)
    GaussianDistributions.conditional(
    Gaussian([x.μ; H*x.μ], [x.Σ x.Σ*H'; H*x.Σ (H*x.Σ*H' + R)]),
    [1,2,3],
    [4,5],
    y)
end
x = Gaussian(μ, Σ)
y = H*rand(x) + rand(Gaussian(zeros(2), R))
z1 = GaussianDistributions.correct(x, (y,R), H)[1]
z2 = correct2(x, Gaussian(y,R), H)
@test z1 ≈ z2
