# Script to solve Burgers' equation numerically to construct PDE solution datasets
# for training and sampling.

using HDF5
using Random

"""
    solve_burgers(p_loc, u_bc; L=1.0, T=1.0, Nx=100, Nt=100, eps=0.02)

Solve 1D Burgers' equation using a Godunov finite-volume scheme.

Returns the solution array of shape (Nx+1, Nt+1).

# Arguments
  - `p_loc`: Location parameter for the sigmoid initial condition
  - `u_bc`: Left boundary condition value
  - `L`, `T`: Domain length and final time
  - `Nx`, `Nt`: Number of spatial and temporal cells
  - `eps`: Sharpness of the initial sigmoid
"""
function solve_burgers(p_loc, u_bc; L=1.0, T=1.0, Nx=100, Nt=100, eps=0.02)
    dx = L / Nx
    dt = T / Nt
    x = range(0.0, L, length=Nx + 1)
    u = zeros(Float64, Nt + 1, Nx + 1)
    u[1, :] .= 1.0 ./ (1.0 .+ exp.((x .- p_loc) ./ eps))  # initial condition

    function godunov_flux(uL, uR)
        flux = zeros(length(uL))
        for i in eachindex(uL)
            if uL[i] <= uR[i]
                # Rarefaction: min of f(uL), f(uR), with f(u)=u²/2
                flux[i] = min(0.5 * uL[i]^2, 0.5 * uR[i]^2)
            else
                # Shock: upwind based on Rankine-Hugoniot speed
                s = (uL[i] + uR[i]) / 2
                flux[i] = s > 0 ? 0.5 * uL[i]^2 : 0.5 * uR[i]^2
            end
        end
        return flux
    end

    for n in 1:Nt
        if n > 1
            u[n, 1] = u_bc
        end
        flux = godunov_flux(u[n, 1:end-1], u[n, 2:end])
        u[n+1, 2:end-1] .= u[n, 2:end-1] .- (dt / dx) .* (flux[2:end] .- flux[1:end-1])
        u[n+1, 1] = u_bc
        u[n+1, end] = u[n+1, end-1]
    end

    # Return (Nx+1, Nt+1) — spatial index first
    return permutedims(u, (2, 1))
end

"""
    generate_burgers_dataset(path, N_ic, N_bc; Nx=100, Nt=100, T=1.0, seed=42, filename="burgers_train")

Generate a Burgers' equation dataset by varying initial and boundary conditions,
and save it to an HDF5 file compatible with the Python h5py format.

# Arguments
  - `path`: Directory in which to save the HDF5 file
  - `N_ic`: Number of initial conditions (random sigmoid locations)
  - `N_bc`: Number of boundary conditions (random left BC values)
  - `Nx`, `Nt`: Spatial and temporal resolution
  - `T`: Final time
  - `seed`: Random seed for reproducibility
  - `filename`: Base filename (without extension)
"""
function generate_burgers_dataset(path, N_ic, N_bc; Nx=100, Nt=100, T=1.0, seed=42, filename="burgers_train")
    Random.seed!(seed)
    p_locs = 0.2 .+ 0.6 .* rand(N_ic)
    u_bcs  = rand(N_bc)
    x = collect(range(0.0, 1.0, length=Nx + 1))
    t = collect(range(0.0, T,   length=Nt + 1))

    mkpath(path)
    full_path = joinpath(path, "$(filename)_nIC$(N_ic)_nBC$(N_bc).h5")

    h5open(full_path, "w") do f
        # Write metadata arrays first
        f["ic"] = Float32.(p_locs)
        f["bc"] = Float32.(u_bcs)
        f["x"]  = Float32.(x)
        f["t"]  = Float32.(t)

        # Pre-allocate the solution dataset.
        # Python/h5py shape: (N_ic, N_bc, Nx+1, Nt+1)
        # Julia HDF5.jl stores in Fortran order, so create with reversed dims:
        d_create(f, "u", datatype(Float32),
                 dataspace(Nt + 1, Nx + 1, N_bc, N_ic))

        Threads.@threads for i_ic in 1:N_ic
            for i_bc in 1:N_bc
                sol = solve_burgers(p_locs[i_ic], u_bcs[i_bc]; Nx=Nx, Nt=Nt, T=T)
                # sol is (Nx+1, Nt+1); store transposed so Python reads (Nx+1, Nt+1)
                f["u"][:, :, i_bc, i_ic] = Float32.(permutedims(sol, (2, 1)))
            end
        end
    end

    println("Saved to $full_path")
    return full_path
end

"""
    generate_burgers_dataset_diffBCs(path; N_bc=20, N_ic=512, Nx=100, Nt=100, T=1.0, seed=42, filename="burgers_sampling_diffBCs")

Generate a Burgers' dataset with the batch dimension organised as (N_bc, N_ic) —
useful for evaluating generalisation over boundary conditions.
"""
function generate_burgers_dataset_diffBCs(
    path; N_bc=20, N_ic=512, Nx=100, Nt=100, T=1.0, seed=42,
    filename="burgers_sampling_diffBCs"
)
    Random.seed!(seed)
    u_bcs  = rand(N_bc)
    p_locs = 0.2 .+ 0.6 .* rand(N_ic)
    x = collect(range(0.0, 1.0, length=Nx + 1))
    t = collect(range(0.0, T,   length=Nt + 1))

    mkpath(path)
    full_path = joinpath(path, "$(filename)_nBC$(N_bc)_nIC$(N_ic).h5")

    h5open(full_path, "w") do f
        f["bc"] = Float32.(u_bcs)
        f["ic"] = Float32.(p_locs)
        f["x"]  = Float32.(x)
        f["t"]  = Float32.(t)

        # Python/h5py shape: (N_bc, N_ic, Nx+1, Nt+1)
        d_create(f, "u", datatype(Float32),
                 dataspace(Nt + 1, Nx + 1, N_ic, N_bc))

        Threads.@threads for i_bc in 1:N_bc
            for i_ic in 1:N_ic
                sol = solve_burgers(p_locs[i_ic], u_bcs[i_bc]; Nx=Nx, Nt=Nt, T=T)
                f["u"][:, :, i_ic, i_bc] = Float32.(permutedims(sol, (2, 1)))
            end
        end
    end

    println("Saved to $full_path")
    return full_path
end

# ── Main script ──────────────────────────────────────────────────────────────

# Training data: vary IC and BC
generate_burgers_dataset("datasets/data/", 80, 80; seed=42, filename="burgers_train")
generate_burgers_dataset("datasets/data/", 30, 30; seed=0,  filename="burgers_test")

# Sampling data for fixed ICs (many BCs per IC)
generate_burgers_dataset("datasets/data/", 20, 512; Nx=100, Nt=100, seed=42,
                         filename="burgers_sampling_diffICs")

# Sampling data for fixed BCs (many ICs per BC)
generate_burgers_dataset_diffBCs("datasets/data/"; N_bc=20, N_ic=512, Nx=100, Nt=100,
                                 seed=42, filename="burgers_sampling_diffBCs")
