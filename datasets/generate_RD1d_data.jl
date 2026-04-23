# Script to solve the Reaction-Diffusion equation numerically to construct PDE
# solution datasets for training and sampling.

using HDF5
using Random

# ── Global parameters ─────────────────────────────────────────────────────────

const rho_RD = 0.01
const nu_RD  = 0.005
const Nx_RD  = 128
const xL_RD, xR_RD = 0.0, 1.0
const dt_save = 0.01
const ini_time = 0.0
const fin_time = 1.0 - dt_save

const dx_RD = (xR_RD - xL_RD) / Nx_RD
const x_RD  = collect(xL_RD + 0.5*dx_RD : dx_RD : xR_RD - 0.5*dx_RD)  # cell-centred grid
const nt_RD = Int(ceil((fin_time - ini_time) / dt_save)) + 1
const t_grid_RD = collect(range(ini_time, fin_time, length=nt_RD))

# ── Solver ────────────────────────────────────────────────────────────────────

"""
    courant_diff(dx; epsilon=1e-3)

Diffusive Courant time-step limit.
"""
courant_diff(dx; epsilon=1e-3) = 0.5 * dx^2 / (epsilon + 1e-8)

"""
    flux_rd(u, gL, gR, dx, N)

Compute the diffusive flux for the reaction-diffusion solver.
"""
function flux_rd(u, gL, gR, dx, N)
    u_ext = zeros(N + 4)
    u_ext[3:N+2] .= u
    f = -nu_RD .* (u_ext[3:N+3] .- u_ext[2:N+2]) ./ dx
    f[1] = gL
    f[end] = gR
    return f
end

"""
    update_rd(u, u_tmp, dt, gL, gR, dx)

One Strang-split half-step: reaction (exact) then diffusion (explicit Euler).
"""
function update_rd(u, u_tmp, dt, gL, gR, dx)
    N = length(u)
    stiff = 1.0 ./ (1.0 .+ exp.(-rho_RD .* dt) .* (1.0 .- u) ./ (u .+ 1e-12))
    f = flux_rd(u_tmp, gL, gR, dx, N)
    return stiff .- dt .* (f[2:N+1] .- f[1:N]) ./ dx
end

"""
    solve_single(u0, gL, gR)

Time-integrate the 1D reaction-diffusion equation with initial condition `u0`
and Neumann-type boundary fluxes `gL` (left) and `gR` (right).

Returns the solution array of shape (Nx, nt).
"""
function solve_single(u0, gL, gR)
    u = copy(u0)
    sol = zeros(Float64, Nx_RD, nt_RD)
    sol[:, 1] .= u
    t = ini_time
    save_idx = 2
    CFL = 0.25

    while t < fin_time && save_idx <= nt_RD
        dt = min(courant_diff(dx_RD; epsilon=nu_RD) * CFL,
                 fin_time - t,
                 t_grid_RD[save_idx] - t)
        dt <= 1e-8 && break
        u_tmp = update_rd(u, u, 0.5 * dt, gL, gR, dx_RD)
        u     = update_rd(u, u_tmp, dt, gL, gR, dx_RD)
        t += dt
        if t >= t_grid_RD[save_idx] - 1e-12
            sol[:, save_idx] .= u
            save_idx += 1
        end
    end
    sol[:, end] .= u
    return sol  # (Nx, nt)
end

# ── Initial condition generator ───────────────────────────────────────────────

"""
    generate_ic(xc; k_tot=3, num_choice_k=2)

Generate a random smooth initial condition on the grid `xc` as a sum of sinusoids,
normalised to [0, 1].
"""
function generate_ic(xc; k_tot=3, num_choice_k=2)
    selected = rand(1:k_tot, num_choice_k)
    onehot = zeros(Int, k_tot)
    for j in selected
        onehot[j] += 1
    end
    kk = 2π .* (1:k_tot) .* onehot ./ (xc[end] - xc[1])
    amp = rand(k_tot, 1)
    phs = 2π .* rand(k_tot, 1)
    u = vec(sum(amp .* sin.(kk .* xc' .+ phs), dims=1))

    if rand() < 0.1
        u = abs.(u)
    end
    u .*= rand([-1, 1])

    if rand() < 0.1
        xL_m = rand() * 0.35 + 0.1
        xR_m = rand() * 0.35 + 0.55
        trns = 0.01
        mask = 0.5 .* (tanh.((xc .- xL_m) ./ trns) .- tanh.((xc .- xR_m) ./ trns))
        u .*= mask
    end

    u .-= minimum(u)
    if maximum(u) > 0
        u ./= maximum(u)
    end
    return u
end

# ── Dataset generator ─────────────────────────────────────────────────────────

"""
    run_parallel(root; N_ic=80, N_bc=80, seed=42, filename="RD_neumann_train")

Generate a reaction-diffusion dataset with `N_ic` initial conditions and
`N_bc` boundary condition pairs, and save to an HDF5 file.

# Arguments
  - `root`: Directory in which to save the HDF5 file
  - `N_ic`: Number of initial conditions
  - `N_bc`: Number of Neumann BC pairs (gL, gR)
  - `seed`: Random seed
  - `filename`: Base filename (without extension)
"""
function run_parallel(root; N_ic=80, N_bc=80, seed=42, filename="RD_neumann_train")
    Random.seed!(seed)
    ic_array = [generate_ic(x_RD) for _ in 1:N_ic]
    bc_array = [(round(0.05 * rand(), digits=3), round(-0.05 * rand(), digits=3))
                for _ in 1:N_bc]

    mkpath(root)
    path = joinpath(root, "$(filename)_nIC$(N_ic)_nBC$(N_bc).h5")

    h5open(path, "w") do f
        # Metadata
        f["ic"] = Float32.(hcat(ic_array...))  # (Nx, N_ic)
        bc_mat  = Float32.(hcat([[bc[1], bc[2]] for bc in bc_array]...))  # (2, N_bc)
        f["bc"] = bc_mat
        f["x"]  = Float32.(x_RD)
        f["t"]  = Float32.(t_grid_RD)
        HDF5.attributes(f)["rho"] = rho_RD
        HDF5.attributes(f)["nu"]  = nu_RD

        # Python/h5py shape: (N_ic, N_bc, Nx, nt)
        # Julia HDF5 reversed dims: (nt, Nx, N_bc, N_ic)
        d_create(f, "u", datatype(Float32),
                 dataspace(nt_RD, Nx_RD, N_bc, N_ic))

        Threads.@threads for i_ic in 1:N_ic
            for i_bc in 1:N_bc
                gL, gR = bc_array[i_bc]
                sol = solve_single(ic_array[i_ic], gL, gR)  # (Nx, nt)
                # Store as (nt, Nx) so Python reads (Nx, nt)
                f["u"][:, :, i_bc, i_ic] = Float32.(permutedims(sol, (2, 1)))
            end
        end
    end

    println("Saved to $path")
    return path
end

# ── Main script ───────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    # Training data: vary IC and BC
    run_parallel("datasets/data/"; N_ic=80, N_bc=80, seed=42, filename="RD_neumann_train")
    run_parallel("datasets/data/"; N_ic=30, N_bc=30, seed=0,  filename="RD_neumann_test")

    # Sampling data: fixed ICs, many BCs
    run_parallel("datasets/data/"; N_ic=20, N_bc=512, seed=42, filename="RD_sampling_diffICs")
end
