# Modifications for PCFM © 2025 Pengfei Cai (Learning Matter @ MIT) and Utkarsh (Julia Lab @ MIT), licensed under the MIT License.
# Original portions © Amazon.com, Inc. or its affiliates, licensed under the Apache License 2.0.

"""
Original code from Cheng et al and also adapted from Li et al, 2020.
Generate random Gaussian fields via spectral sampling.
"""

"""
    GaussianRF

Gaussian random field sampler using spectral (Fourier) representation.

Supports 1D, 2D, and 3D fields. Samples are drawn by generating random
Fourier coefficients scaled by the square-root eigenvalues of the covariance
operator, then applying an unnormalized inverse FFT.

# Fields
  - `n_dims`: Number of spatial dimensions (1, 2, or 3)
  - `sqrt_eig`: Spectral scaling weights (same shape as the grid)
  - `sz`: Grid size tuple, e.g. `(size,)`, `(size, size)`, or `(size, size, size)`
"""
struct GaussianRF
    n_dims::Int
    sqrt_eig::Array{Float64}
    sz::NTuple{<:Any, Int}
end

"""
    GaussianRF(n_dims, size; alpha=2, tau=3, sigma=nothing)

Construct a `GaussianRF` for a uniform grid of side length `size` in `n_dims` dimensions.

# Arguments
  - `n_dims`: Number of spatial dimensions (1, 2, or 3)
  - `size`: Number of grid points per dimension
  - `alpha`: Regularity parameter (default 2)
  - `tau`: Length-scale parameter (default 3)
  - `sigma`: Scaling constant; defaults to `tau^(0.5*(2*alpha - n_dims))`
"""
function GaussianRF(n_dims::Int, sz::Int; alpha::Real=2, tau::Real=3, sigma=nothing)
    if sigma === nothing
        sigma = tau^(0.5 * (2 * alpha - n_dims))
    end

    k_max = sz ÷ 2

    if n_dims == 1
        k = Float64.([0:(k_max - 1); -k_max:-1])
        sqrt_eig = sz * sqrt(2.0) * sigma .*
                   (4 * π^2 .* k .^ 2 .+ tau^2) .^ (-alpha / 2.0)
        sqrt_eig[1] = 0.0

    elseif n_dims == 2
        k = Float64.([0:(k_max - 1); -k_max:-1])
        k_x = reshape(k, sz, 1)   # varies along first dim
        k_y = reshape(k, 1, sz)   # varies along second dim
        sqrt_eig = (sz^2) * sqrt(2.0) * sigma .*
                   (4 * π^2 .* (k_x .^ 2 .+ k_y .^ 2) .+ tau^2) .^ (-alpha / 2.0)
        sqrt_eig[1, 1] = 0.0

    elseif n_dims == 3
        k = Float64.([0:(k_max - 1); -k_max:-1])
        k_x = reshape(k, sz, 1, 1)
        k_y = reshape(k, 1, sz, 1)
        k_z = reshape(k, 1, 1, sz)
        sqrt_eig = (sz^3) * sqrt(2.0) * sigma .*
                   (4 * π^2 .* (k_x .^ 2 .+ k_y .^ 2 .+ k_z .^ 2) .+ tau^2) .^ (-alpha / 2.0)
        sqrt_eig[1, 1, 1] = 0.0

    else
        error("n_dims must be 1, 2, or 3")
    end

    return GaussianRF(n_dims, sqrt_eig, ntuple(_ -> sz, n_dims))
end

"""
    sample(grf::GaussianRF, N::Int)

Draw `N` independent samples from the Gaussian random field.

Returns an array of shape `(grf.sz..., N)` where `grf.sz` is the spatial grid size.

# Example

```julia
grf = GaussianRF(2, 64; alpha=2.5, tau=7)
u = sample(grf, 10)  # (64, 64, 10)
```
"""
function sample(grf::GaussianRF, N::Int)
    # Generate complex Gaussian random coefficients: (sz..., N)
    coeff = randn(ComplexF64, grf.sz..., N)

    # Scale each spatial mode by sqrt_eig (broadcast over batch dimension)
    coeff .*= grf.sqrt_eig

    # Unnormalized backward IFFT over spatial dims (matches torch norm='backward')
    spatial_dims = 1:grf.n_dims
    u = real.(bfft(coeff, spatial_dims))

    return u
end
