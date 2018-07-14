# GaussianDistributions

To install run `Pkg.add("GaussianDistributions")`.

This package creates a simplistic alternative to `Distributions` to generate Gaussian or normally distributed random variables and deal with their distributions. 

The *raison d’être* of GaussianDistributions is that the type hierarchy layed out by distributions is not well suited to handle singular covariance matrices, and anything other than `Float64`s and `Vector{Float64}`s.

This package contains enough functionality such that `Gaussian`s can be useful as Gaussian state variable for example in a Kalman
filter.

It also provides cumulative distributions functions (`cdf`) for bivariate Normal distributions using an approximation (valid unless the correlation `ρ` is very strong and `x ≈ y*sign(ρ)`.)