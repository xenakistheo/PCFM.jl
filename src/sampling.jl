"""
    sample_ffm(ffm::FFM, tstate, n_samples, n_steps;
               use_compiled=false, compiled_funcs=nothing, verbose=true)

Generate samples from the trained Functional Flow Matching model using Euler integration.

# Arguments

  - `ffm`: FFM model
  - `tstate`: Training state (or use `ffm.ps` and `ffm.st` directly)
  - `n_samples`: Number of samples to generate
  - `n_steps`: Number of Euler integration steps
  - `use_compiled`: Whether to use compiled functions
  - `compiled_funcs`: Compiled functions from `compile_functions`
  - `verbose`: Print progress

# Returns

  - Generated samples of shape (spatial_size..., nt, 1, n_samples)

# Example

```julia
samples = sample_ffm(ffm, tstate, 32, 100)
```
"""
function sample_ffm(ffm::FFM, tstate, n_samples, n_steps;
        use_compiled = true,
        compiled_funcs = nothing,
        verbose = true,
        initial_vals=nothing)

    spatial_size = ffm.config[:spatial_size]
    nt           = ffm.config[:nt]
    emb_channels = ffm.config[:emb_channels]
    device       = ffm.config[:device]

    if hasfield(typeof(tstate), :parameters)
        ps = tstate.parameters
        st = tstate.states
    else
        ps = tstate[1]
        st = tstate[2]
    end

    if use_compiled && compiled_funcs !== nothing
        model_fn         = compiled_funcs.model
        prepare_input_fn = compiled_funcs.prepare_input
    else
        model_fn         = ffm.model
        prepare_input_fn = prepare_input
    end

    if initial_vals !== nothing
        @assert size(initial_vals) == (spatial_size..., nt, 1, n_samples)
        x = initial_vals |> device
    else
        x = randn(Float32, spatial_size..., nt, 1, n_samples) |> device
    end

    dt = 1.0f0 / n_steps

    for step in 0:(n_steps - 1)
        verbose && step % 10 == 0 && println("Sampling step: $step/$n_steps")

        t_vec   = fill(Float32(step * dt), n_samples) |> device
        x_input = prepare_input_fn(x, t_vec, spatial_size, nt, n_samples, emb_channels)
        v, st   = model_fn(x_input, ps, st)
        x       = x .+ v .* dt
    end

    return x
end

"""
    sample_pcfm(ffm::FFM, tstate, n_samples, n_steps;
                use_compiled=false, compiled_funcs=nothing, verbose=true)

Generate physics-constrained samples using PCFM algorithm.
Fixed initial condition: u(x,0) = sin(x + π/4)

# Arguments
  - `ffm`: FFM model
  - `tstate`: Training state
  - `n_samples`: Number of samples to generate
  - `n_steps`: Number of integration steps
  - `use_compiled`: Whether to use compiled functions
  - `compiled_funcs`: Compiled functions
  - `verbose`: Print progress

# Returns
  - Generated samples satisfying the initial condition constraint
"""
function sample_pcfm(ffm::FFM, tstate, n_samples, n_steps;
        use_compiled = true,
        compiled_funcs = nothing,
        verbose = true)
    spatial_size = ffm.config[:spatial_size]
    nx           = spatial_size[1]   # 1D-specific constraint sampling
    nt           = ffm.config[:nt]
    emb_channels = ffm.config[:emb_channels]
    device       = ffm.config[:device]

    if hasfield(typeof(tstate), :parameters)
        ps = tstate.parameters
        st = tstate.states
    else
        ps = tstate[1]
        st = tstate[2]
    end

    if use_compiled && compiled_funcs !== nothing
        model_fn         = compiled_funcs.model
        prepare_input_fn = compiled_funcs.prepare_input
    else
        model_fn         = ffm.model
        prepare_input_fn = prepare_input
    end

    # Fixed initial condition: u(x,0) = sin(x + π/4)
    x_grid = range(0, 2π, length=nx)
    u_0_ic = sin.(x_grid .+ π/4)
    u_0_ic = repeat(reshape(u_0_ic, nx, 1, 1, 1), 1, 1, 1, n_samples) |> device

    x_0 = randn(Float32, spatial_size..., nt, 1, n_samples) |> device
    x   = copy(x_0)
    dt  = 1.0f0 / n_steps

    for step in 0:(n_steps - 1)
        verbose && step % 10 == 0 && println("PCFM step: $step/$n_steps")

        τ      = step * dt
        τ_next = τ + dt
        t_vec  = fill(Float32(τ), n_samples) |> device

        x_input = prepare_input_fn(x, t_vec, spatial_size, nt, n_samples, emb_channels)
        v, st   = model_fn(x_input, ps, st)

        x_1 = x .+ v .* (1.0f0 - τ)
        @. x_1[:, 1:1, :, :] = u_0_ic

        x = x_0 .+ (x_1 .- x_0) .* τ_next
    end

    return x
end


"""
    sample_pcfm(ffm::FFM, tstate, n_samples, n_steps, H!; ...)

Generate physics-constrained samples using PCFM. Dimension-agnostic: works for any
spatial dimensionality supported by the FFM model (1D, 2D, ...).

# Arguments
  - `ffm`: FFM model
  - `tstate`: Training state
  - `n_samples`: Number of samples to generate
  - `n_steps`: Number of integration steps
  - `H!`: Constraint function (dispatches to JuMP or ExaModels variant)
  - `IC_func`: 1D only — scalar `x -> scalar` initial condition function
  - `IC_field`: Any dim — precomputed IC array of shape `(spatial_size...)`; takes precedence over `IC_func`
  - `domain`: NamedTuple with at least `x_start`, `x_end` fields; optionally `y_start`, `y_end` for 2D
  - `backend`: KernelAbstractions backend (default `CPU()`)
  - `mode`: `"exa"` (ExaModels + MadNLP) or `"jump"` (JuMP + optimizer)
  - `optimizer`: JuMP-compatible optimizer factory (only used when `mode="jump"`)

# Returns
  - Generated samples of shape `(spatial_size..., nt, 1, n_samples)` as a CPU Array
"""
function sample_pcfm(ffm::FFM, tstate, n_samples, n_steps, H! = nothing;
        constraint_parameters = nothing,
        domain = (x_start=0.0f0, x_end=2f0π, t_start=0.0f0, t_end=1.0f0),
        IC_func  = x -> sin(x + π/4),  # 1D only: scalar -> scalar
        IC_field = nothing,             # any dim: precomputed array of shape (spatial_size...)
        backend = CPU(),
        mode = "exa",
        optimizer = MadNLP.Optimizer,
        proj! = nothing,                # analytical projection function (mode="analytical" only)
        use_compiled = true,
        compiled_funcs = nothing,
        verbose = true,
        initial_vals = nothing)

    @assert mode ∈ ("exa", "jump", "analytical") "mode must be \"exa\", \"jump\", or \"analytical\""
    if mode ∈ ("exa", "jump")
        H! !== nothing || error("H! is required for mode=\"$mode\"")
    end
    if mode == "analytical"
        proj! !== nothing || error("proj! is required for mode=\"analytical\"")
    end

    spatial_size = ffm.config[:spatial_size]
    ndim_s       = length(spatial_size)
    nt           = ffm.config[:nt]
    emb_channels = ffm.config[:emb_channels]
    device       = ffm.config[:device]

    println("\n------------------------")
    println("------Sampling PCFM------")
    println("Mode: $mode")
    mode ∈ ("exa", "jump") && println("Optimizer: ", string(optimizer))
    println("Backend: ", string(backend))
    println("------------------------\n")

    if hasfield(typeof(tstate), :parameters)
        ps = tstate.parameters
        st = tstate.states
    else
        ps = tstate[1]
        st = tstate[2]
    end

    if use_compiled && compiled_funcs !== nothing
        model_fn         = compiled_funcs.model
        prepare_input_fn = compiled_funcs.prepare_input
    else
        model_fn         = ffm.model
        prepare_input_fn = prepare_input
    end

    # --- Initial condition: build array of shape (spatial_size...) ---
    if IC_field !== nothing
        u_0_ic_arr = Float32.(IC_field)
        @assert size(u_0_ic_arr) == Tuple(spatial_size) "IC_field must have shape $(Tuple(spatial_size))"
    elseif ndim_s == 1
        x_grid     = range(domain.x_start, domain.x_end; length = spatial_size[1])
        u_0_ic_arr = Float32.(IC_func.(x_grid))
    else
        error("For ndim_s > 1 pass IC_field (array of shape spatial_size = $spatial_size)")
    end

    # Replicate IC across samples: (spatial_size...) -> (spatial_size..., n_samples)
    u_0_ic_mat = KernelAbstractions.adapt(backend,
        repeat(reshape(u_0_ic_arr, spatial_size..., 1);
               outer = (ntuple(_ -> 1, ndim_s)..., n_samples)))

    # --- Grid points and spacing (tuples, one entry per spatial dim) ---
    dx = Float32(domain.x_end - domain.x_start) / (spatial_size[1] - 1)
    if ndim_s == 1
        grid_points  = (spatial_size[1],)
        grid_spacing = (dx,)
    elseif ndim_s == 2
        y_start = hasproperty(domain, :y_start) ? Float32(domain.y_start) : Float32(domain.x_start)
        y_end_v = hasproperty(domain, :y_end)   ? Float32(domain.y_end)   : Float32(domain.x_end)
        dy      = (y_end_v - y_start) / (spatial_size[2] - 1)
        grid_points  = (spatial_size[1], spatial_size[2])
        grid_spacing = (dx, Float32(dy))
    else
        error("Only 1D and 2D spatial problems are currently supported")
    end

    dt = 1.0f0 / n_steps
    N  = prod(spatial_size) * nt * n_samples   # flat size (no channel dim)

    if initial_vals !== nothing
        @assert size(initial_vals) == (spatial_size..., nt, 1, n_samples)
        x_0 = initial_vals |> device
    else
        x_0 = randn(Float32, spatial_size..., nt, 1, n_samples) |> device
    end
    x     = copy(x_0)
    t_vec = fill(0f0, n_samples) |> device

    # --- Build Exa NLP once (sparsity pattern is fixed across steps) ---
    if mode == "exa"
        x1_param = KernelAbstractions.adapt(backend, zeros(Float32, N))
        core     = ExaCore(backend = backend)
        θ        = parameter(core, x1_param)
        u_exa    = variable(core, 1:N; start = x1_param)
        objective(core, (u_exa[i] - θ[i])^2 for i in 1:N)
        H!(core, u_exa, u_0_ic_mat, nt, n_samples, grid_points, grid_spacing, dt,
           constraint_parameters; backend = backend)
        nlp = ExaModel(core)
        solver = if backend isa GPU
            MadNLP.MadNLPSolver(nlp; linear_solver = MadNLPGPU.CUDSSSolver, print_level = MadNLP.ERROR)
        else
            MadNLP.MadNLPSolver(nlp; print_level = MadNLP.ERROR)
        end
    end

    # ND shape for JuMP variable: (spatial_size..., nt, n_samples) — no channel dim
    all_dims = (spatial_size..., nt, n_samples)

    # --- Euler integration ---
    for step in 0:(n_steps - 1)
        verbose && step % 10 == 0 && println("PCFM step: $step/$n_steps")

        τ      = step * dt
        τ_next = τ + dt
        fill!(t_vec, τ)

        x_input = prepare_input_fn(x, t_vec, spatial_size, nt, n_samples, emb_channels)
        v, st   = model_fn(x_input, ps, st)

        # Extrapolate to t=1
        x_1 = x .+ v .* (1.0f0 - τ)

        # Apply physics constraints
        if mode == "analytical"
            x_1 = proj!(x_1, u_0_ic_mat, nt, n_samples, grid_points, grid_spacing, dt,
                        constraint_parameters)
        elseif mode == "jump"
            # x_1 shape: (spatial_size..., nt, 1, n_samples)
            # N = prod(spatial_size)*nt*n_samples, so flatten drops the trivial channel dim
            x_1_flat = reshape(Array(x_1), N)

            model_jump = Model(optimizer)
            set_silent(model_jump)

            # Flat JuMP variable; reshape to ND for the constraint function
            @variable(model_jump, u_flat_jmp[1:N])
            u_nd = reshape([u_flat_jmp[k] for k in 1:N], all_dims)

            @objective(model_jump, Min, sum((u_flat_jmp[k] - x_1_flat[k])^2 for k in 1:N))
            H!(model_jump, u_nd, u_0_ic_mat, nt, n_samples,
               grid_points, grid_spacing, dt, constraint_parameters)
            optimize!(model_jump)

            x_1 = reshape(Float32.(Array(value.(u_flat_jmp))), spatial_size..., nt, 1, n_samples) |> device
        else  # "exa"
            copyto!(nlp.θ, reshape(x_1, N))
            result = MadNLP.solve!(solver)
            x_1    = reshape(Float32.(solution(result, u_exa)), spatial_size..., nt, 1, n_samples) |> device
        end

        @. x = x_0 + (x_1 - x_0) * τ_next
    end

    return Array(x)
end
