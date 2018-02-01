# GaussianDistributions

This package creates an alternative to `Distributions` to generate Gaussian or normally distributed random variables and deal with their distributions. 

The *raison d’être* of GaussianDistributions is that the type hierarchy layed out by distributions is not well suited to handle singular covariance matrices, and anything other than `Float64`s and `Vector{Float64}`s.