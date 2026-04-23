# Modifications for PCFM © 2025 Pengfei Cai (Learning Matter @ MIT) and Utkarsh (Julia Lab @ MIT), licensed under the MIT License.
# Original portions © Amazon.com, Inc. or its affiliates, licensed under the Apache License 2.0.

"""
Original code from Cheng et al and also adapted from Li et al, 2020.

Solve 2D Navier-Stokes equations (vorticity form) with the Crank-Nicolson method
using spectral (pseudo-spectral) spatial discretisation.
"""

using HDF5
using FFTW
using Random
using LinearAlgebra
using CUDA

# Use all available CPU threads for FFTW (no-op on GPU)
FFTW.set_num_threads(Sys.CPU_THREADS)

include("random_fields.jl")

# ── NS solver ─────────────────────────────────────────────────────────────────

"""
    solve_navier_stokes_2d(w0, f; visc=1e-3, T=49, delta_t=1e-3, record_steps=50)

Solve the 2D incompressible Navier-Stokes equations in vorticity form on a
periodic domain using a pseudo-spectral Crank-Nicolson scheme.

Works on CPU (Array) or GPU (CuArray) — spectral constant arrays are moved
to match the device of the input automatically.

# Arguments
  - `w0`: Initial vorticity field, shape `(s, s, batch)`
  - `f`: Forcing term, shape `(s, s)` or `(s, s, batch)`
  - `visc`: Kinematic viscosity (1/Re)
  - `T`: Final time
  - `delta_t`: Internal time-step (decrease if blow-up occurs)
  - `record_steps`: Number of in-time snapshots to record (including t=0)

# Returns
  - Solution array of shape `(s, s, batch, record_steps)` on CPU
"""
function solve_navier_stokes_2d(w0, f; visc=1e-3, T=49, delta_t=1e-3, record_steps=50, verbose=false)
    s     = size(w0, 1)
    batch = size(w0, 3)
    k_max = s ÷ 2
    steps = ceil(Int, T / delta_t)

    on_gpu = w0 isa CuArray

    # Forward FFT over spatial dims (1,2); result shape (s, s, batch)
    w_h = fft(w0, 1:2)

    # Forcing to Fourier space
    f_h = ndims(f) == 2 ? reshape(fft(f, 1:2), s, s, 1) : fft(f, 1:2)

    record_time = steps ÷ (record_steps - 1)

    # Wavenumbers: [0, 1, …, k_max-1, -k_max, …, -1]
    k1d  = Float64.([0:k_max-1; -k_max:-1])
    k_x  = reshape(k1d, s, 1, 1)   # varies along dim 1
    k_y  = reshape(k1d, 1, s, 1)   # varies along dim 2

    # Negative Laplacian in Fourier space
    lap       = 4π^2 .* (k_x .^ 2 .+ k_y .^ 2)
    lap[1, 1, 1] = 1.0  # avoid division by zero for the mean mode

    # Dealiasing mask (2/3 rule)
    dealias = Float64.(abs.(k_x) .<= (2/3) * k_max .&& abs.(k_y) .<= (2/3) * k_max)

    # Move spectral constants to same device as input
    if on_gpu
        k_x     = CuArray(k_x)
        k_y     = CuArray(k_y)
        lap     = CuArray(lap)
        dealias = CuArray(dealias)
    end

    # Solution stored on CPU regardless of compute device
    sol   = zeros(Float64, s, s, batch, record_steps)
    sol[:, :, :, 1] .= on_gpu ? Array(w0) : w0

    c = 2
    t = 0.0

    for j in 1:steps
        # Stream function ψ from Poisson: ψ_h = w_h / (-Δ)
        psi_h = w_h ./ lap

        # Velocity u = ∂ψ/∂y  (in Fourier: multiply by i*k_y)
        q_h      = complex.(-2π .* k_y .* imag.(psi_h),
                              2π .* k_y .* real.(psi_h))
        q        = real.(ifft(q_h, 1:2))

        # Velocity v = -∂ψ/∂x  (in Fourier: multiply by -i*k_x)
        v_h      = complex.( 2π .* k_x .* imag.(psi_h),
                             -2π .* k_x .* real.(psi_h))
        v        = real.(ifft(v_h, 1:2))

        # Partial derivatives of vorticity
        wx_h     = complex.(-2π .* k_x .* imag.(w_h),
                              2π .* k_x .* real.(w_h))
        w_x      = real.(ifft(wx_h, 1:2))

        wy_h     = complex.(-2π .* k_y .* imag.(w_h),
                              2π .* k_y .* real.(w_h))
        w_y      = real.(ifft(wy_h, 1:2))

        # Non-linear term in Fourier space + dealiasing
        F_h = fft(q .* w_x .+ v .* w_y, 1:2) .* dealias

        # Crank-Nicolson update
        factor = 0.5 .* delta_t .* visc .* lap
        num    = -delta_t .* F_h .+ delta_t .* f_h .+ (1.0 .- factor) .* w_h
        w_h    = num ./ (1.0 .+ factor)

        t += delta_t

        if verbose && j % max(1, steps ÷ 10) == 0
            pct = round(Int, 100 * j / steps)
            println("    step $j / $steps  ($pct%)")
            flush(stdout)
        end

        if j % record_time == 0 && c <= record_steps
            w = real.(ifft(w_h, 1:2))
            w_cpu = on_gpu ? Array(w) : w
            if any(isnan, w_cpu)
                error("NaN values encountered at step $j.")
            end
            sol[:, :, :, c] .= w_cpu
            c += 1
        end
    end

    return sol  # (s, s, batch, record_steps) on CPU
end

# ── Main generation function ──────────────────────────────────────────────────

"""
    navier_stokes(root; nw=100, nf=100, s=64, T=49, steps=50, mu=1e-3,
                  batch_size=1024, seed=42, delta=1e-3, use_gpu=false)

Generate a 2D Navier-Stokes dataset with `nw` initial vorticity fields and
`nf` forcing fields, and save to an HDF5 file.

# Arguments
  - `root`: Directory in which to save the HDF5 file
  - `nw`: Number of initial vorticity fields
  - `nf`: Number of forcing fields
  - `s`: Spatial grid size (s × s)
  - `T`: Final time
  - `steps`: Number of recorded time snapshots
  - `mu`: Viscosity
  - `batch_size`: Number of (w, f) pairs to solve simultaneously
  - `seed`: Random seed
  - `delta`: Internal solver time-step
  - `use_gpu`: Move batches to GPU (CuArray) before solving; uses cuFFT automatically
"""
function navier_stokes(root; nw=100, nf=100, s=64, T=49, steps=50, mu=1e-3,
                       batch_size=1024, seed=42, delta=1e-4, use_gpu=false)
    if use_gpu && !CUDA.functional()
        @warn "use_gpu=true but CUDA is not functional — falling back to CPU."
        use_gpu = false
    end

    Random.seed!(seed)

    mkpath(root)
    path = joinpath(root, "ns_nw$(nw)_nf$(nf)_s$(s)_t$(steps)_mu$(mu).h5")

    # Gaussian random field for initial vorticity
    grf = GaussianRF(2, s; alpha=2.5, tau=7)

    # Sample initial vorticity fields: (s, s, nw)
    w0 = Float32.(sample(grf, nw))

    # Deterministic sinusoidal forcing: phase varies with forcing index
    ft  = range(0.0, 1.0, length=s+1)[1:end-1]
    X   = [x for x in ft, _ in ft]   # (s, s)
    Y   = [y for _ in ft, y in ft]   # (s, s)
    phi = Float32.(π/2 .* range(0.0, 1.0, length=nf))  # (nf,)
    fs  = Float32.(0.1 * sqrt(2) .* sin.(2π .* (X .+ Y) .+ reshape(phi, 1, 1, nf)))
    # shape: (s, s, nf)

    h5open(path, "w") do f
        # Python/h5py shapes: a:(nw,s,s), f:(nf,s,s), u:(nw,nf,s,s,steps)
        # Julia HDF5 reversed dims: a:(s,s,nw), f:(s,s,nf), u:(steps,s,s,nf,nw)
        create_dataset(f, "a", datatype(Float32), dataspace(s, s, nw))
        create_dataset(f, "f", datatype(Float32), dataspace(s, s, nf))
        create_dataset(f, "u", datatype(Float32), dataspace(steps, s, s, nf, nw))

        f["a"][:, :, :] = w0
        f["f"][:, :, :] = fs

        # Flatten to (s, s, nw*nf) pairs for batched solving.
        # Index layout: global index k = (i_w - 1)*nf + i_f
        w0_exp = reshape(repeat(reshape(w0, s, s, nw, 1), outer=(1,1,1,nf)), s, s, nw*nf)
        fs_exp = reshape(repeat(reshape(fs, s, s, 1, nf), outer=(1,1,nw,1)), s, s, nw*nf)

        n_total = nw * nf
        n_batch = ceil(Int, n_total / batch_size)

        for b in 1:n_batch
            idx_start = (b - 1) * batch_size + 1
            idx_end   = min(b * batch_size, n_total)
            batch_len = idx_end - idx_start + 1
            println("Batch $b / $n_batch  (samples $(idx_start)-$(idx_end), size=$(batch_len))")
            flush(stdout)

            w_batch = Float64.(w0_exp[:, :, idx_start:idx_end])
            f_batch = Float64.(fs_exp[:, :, idx_start:idx_end])

            # Optionally move to GPU — fft/ifft dispatch to cuFFT automatically
            if use_gpu
                w_batch = CuArray(w_batch)
                f_batch = CuArray(f_batch)
            end

            t_batch = @elapsed sol = solve_navier_stokes_2d(w_batch, f_batch;
                                         visc=mu, T=T, delta_t=delta,
                                         record_steps=steps, verbose=true)
            println("  Batch $b done in $(round(t_batch/60, digits=1)) min")
            flush(stdout)
            # sol is always on CPU (Array); write directly to HDF5
            for (local_idx, global_idx) in enumerate(idx_start:idx_end)
                i_w = (global_idx - 1) ÷ nf + 1
                i_f = (global_idx - 1) % nf + 1
                f["u"][:, :, :, i_f, i_w] = Float32.(permutedims(sol[:, :, local_idx, :], (3, 1, 2)))
            end
        end
    end

    println("Done. Dataset saved to $path")
    return path
end

# ── Main script ───────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    navier_stokes("datasets/data/";
                  nw=100, nf=100, s=64, T=49, steps=50,
                  mu=1e-3, batch_size=1024, seed=42, delta=1e-4)
end
