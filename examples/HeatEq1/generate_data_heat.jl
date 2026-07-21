# Script to construct 1D heat (diffusion) equation solution datasets for
# training and sampling. The heat equation has a closed-form solution here,
# so "solving" it is just evaluating sin(x+φ)·exp(-tν) at random (ν, φ) pairs
# rather than numerically integrating a PDE.

using HDF5
using Random

"""
    generate_diffusion_data(n_samples, nx, nt, visc_range, phi_range, t_range)

Generate 1D diffusion dataset (space-time).

# Arguments

  - `n_samples`: Number of samples to generate
  - `nx`: Number of spatial points
  - `nt`: Number of temporal points
  - `visc_range`: Tuple of (min, max) viscosity values
  - `phi_range`: Tuple of (min, max) phase shift values
  - `t_range`: Tuple of (start, end) time range

# Returns

  - `u_data`: Array of shape (nx, nt, 1, n_samples) containing the diffusion solutions

# Example

```julia
u_data = generate_diffusion_data(
    32, 100, 100, (1.0f0, 5.0f0), (0.0f0, Float32(π)), (0.0f0, 1.0f0))
```

The data follows the analytical solution: u(x,t) = sin(x + φ) * exp(-t * ν)
"""
function generate_diffusion_data(n_samples, nx, nt, visc_range, phi_range, t_range)
    xs = range(0.0f0, 2.0f0 * Float32(π), length = nx+1)[1:(end - 1)]
    ts = range(t_range[1], t_range[2], length = nt)

    # Julia format: (nx, nt, 1, n_samples) = (H, W, C, B)
    u_data = zeros(Float32, nx, nt, 1, n_samples)

    for i in 1:n_samples
        v = visc_range[1] + rand(Float32) * (visc_range[2] - visc_range[1])
        phi = phi_range[1] + rand(Float32) * (phi_range[2] - phi_range[1])

        # u(x,t) = sin(x + phi) * exp(-t * v)
        for (ti, t) in enumerate(ts)
            u_data[:, ti, 1, i] .= sin.(xs .+ phi) .* exp(-t * v)
        end
    end

    return u_data
end

"""
    generate_heat_dataset(path, n_samples; nx=100, nt=100,
                           visc_range=(1.0f0, 5.0f0), phi_range=(0.0f0, Float32(π)),
                           t_range=(0.0f0, 1.0f0), seed=42, filename="heat_train")

Generate a 1D heat-equation dataset of `n_samples` analytic solutions with
random (viscosity, phase) pairs, and save it to an HDF5 file compatible with
the Python h5py format.

# Arguments
  - `path`: Directory in which to save the HDF5 file
  - `n_samples`: Number of (viscosity, phase) solutions to generate
  - `nx`, `nt`: Spatial and temporal resolution
  - `visc_range`, `phi_range`, `t_range`: Sampling ranges (see `generate_diffusion_data`)
  - `seed`: Random seed for reproducibility
  - `filename`: Base filename (without extension)
"""
function generate_heat_dataset(path, n_samples; nx=100, nt=100,
                                visc_range=(1.0f0, 5.0f0),
                                phi_range=(0.0f0, Float32(π)),
                                t_range=(0.0f0, 1.0f0),
                                seed=42, filename="heat_train")
    Random.seed!(seed)
    xs = collect(range(0.0f0, 2.0f0 * Float32(π), length=nx + 1)[1:(end - 1)])
    ts = collect(range(t_range[1], t_range[2], length=nt))
    viscs = visc_range[1] .+ rand(Float32, n_samples) .* (visc_range[2] - visc_range[1])
    phis  = phi_range[1]  .+ rand(Float32, n_samples) .* (phi_range[2]  - phi_range[1])

    mkpath(path)
    full_path = joinpath(path, "$(filename)_n$(n_samples).h5")

    h5open(full_path, "w") do f
        f["visc"] = viscs
        f["phi"]  = phis
        f["x"]    = Float32.(xs)
        f["t"]    = Float32.(ts)

        # Python/h5py shape: (n_samples, Nx, Nt)
        # Julia HDF5.jl stores in Fortran order, so create with reversed dims:
        create_dataset(f, "u", datatype(Float32),
                 dataspace(nt, nx, n_samples))

        Threads.@threads for i in 1:n_samples
            u = sin.(xs .+ phis[i]) .* exp.(-ts' .* viscs[i])  # (nx, nt)
            f["u"][:, :, i] = Float32.(permutedims(u, (2, 1)))
        end
    end

    println("Saved to $full_path")
    return full_path
end

# ── Main script ──────────────────────────────────────────────────────────────
#
# Generate 1D heat (diffusion) equation training and test datasets and save
# to HDF5.
#
# Grid: Nx=100, Nt=100, Ω=[0,2π], T=1. ν (viscosity) ∈ [1,5], φ (phase) ∈ [0,π].
#
# Output files (written to datasets/data/):
#   - heat_train_n6400.h5   (train: 6400 samples)
#   - heat_test_n900.h5     (test:   900 samples)

if abspath(PROGRAM_FILE) == @__FILE__
    data_dir = joinpath(@__DIR__, "..", "..", "datasets", "data")

    println("Generating heat training set (6400 samples)...")
    generate_heat_dataset(data_dir, 6400; seed=42, filename="heat_train")

    println("Generating heat test set (900 samples)...")
    generate_heat_dataset(data_dir, 900; seed=0, filename="heat_test")

    println("Done.")
end
