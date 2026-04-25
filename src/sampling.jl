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

  - Generated samples of shape (nx, nt, 1, n_samples)

# Example

```julia
samples = sample_ffm(ffm, tstate, 32, 100)
```
"""
function sample_ffm(ffm::FFM, tstate, n_samples, n_steps;
        use_compiled = true,
        compiled_funcs = nothing,
        verbose = true)
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

    x  = randn(Float32, spatial_size..., nt, 1, n_samples) |> device
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
function sample_pcfm(ffm::FFM, tstate, n_samples, n_steps, H!, params;
        backend = CPU(),
        mode = "exa",
        use_compiled = true,
        compiled_funcs = nothing,
        verbose = true)

    spatial_size = ffm.config[:spatial_size]
    nx           = spatial_size[1]   # 1D-specific constrained sampling
    nt           = ffm.config[:nt]
    emb_channels = ffm.config[:emb_channels]
    device       = ffm.config[:device]

    @show backend isa GPU

    # Extract parameters and states
    if hasfield(typeof(tstate), :parameters)
        ps = tstate.parameters
        st = tstate.states
    else
        ps = tstate[1]
        st = tstate[2]
    end

    # Use compiled or regular functions
    if use_compiled && compiled_funcs !== nothing
        model_fn = compiled_funcs.model
        prepare_input_fn = compiled_funcs.prepare_input
    else
        model_fn = ffm.model
        prepare_input_fn = prepare_input
    end

    # Fixed initial condition: u(x,0) = sin(x + π/4)
    x_grid = range(0, 2π, length=nx)
    u_0_ic = sin.(x_grid .+ π/4)
    u_0_ic = reshape(u_0_ic, nx, 1, 1, 1)
    # Broadcast to all samples
    u_0_ic = repeat(u_0_ic, 1, 1, 1, n_samples) |> device

    # Start from Gaussian noise
    x_0 = randn(Float32, spatial_size..., nt, 1, n_samples) |> device
    x = copy(x_0)

    dt = 1.0f0 / n_steps
    dx = x_grid[2] - x_grid[1]

    # Euler integration from t=0 to t=1
    for step in 0:(n_steps - 1)
        if verbose && step % 10 == 0
            println("PCFM step: $step/$n_steps")
        end

        τ = step * dt
        τ_next = τ + dt
        t_vec = fill(Float32(τ), n_samples) |> device

        # Prepare input with embeddings
        x_input = prepare_input_fn(x, t_vec, spatial_size, nt, n_samples, emb_channels)

        # Predict velocity field
        v, st = model_fn(x_input, ps, st)

        # Step 1: Extrapolate to t=1
        x_1 = x .+ v .* (1.0f0 - τ)

        # Step 2: Apply constraint - fix initial condition
        ##############
        if mode == "jump"
            # JuMP version — batched over all samples at once
            x_1_cpu = Array(x_1)  # (nx, nt, 1, n_samples)
            model = Model(MadNLP.Optimizer)
            set_silent(model)
            @variable(model, u[1:nx, 1:nt, 1:n_samples])
            @objective(model, Min, sum((u[i,j,s] - x_1_cpu[i,j,1,s])^2 for i in 1:nx, j in 1:nt, s in 1:n_samples))
            @constraint(model, [i in 1:nx, s in 1:n_samples], u[i, 1, s] == x_1_cpu[i, 1, 1, s])
            @constraint(model, [j in 1:nt, s in 1:n_samples], dx * sum(u[i,j,s] for i in 1:(nx-1)) == 0.0)
            optimize!(model)
            x_0 = reshape(Float32.(value.(u)), nx, nt, 1, n_samples) |> device
        else
            # ExaModel version — solve projection for all samples at once
            N = nx * nt * n_samples

            if backend isa GPU
                x_1_b = x_1[:, :, 1, :]
                x1_vec = KernelAbstractions.adapt(backend, vec(x_1_b))
                core = ExaCore(backend=backend)
                θ = parameter(core, x1_vec)              # x_1_flat can be a CuArray
                u = variable(core, 1:N; start = x1_vec)
                objective(core, (u[i] - θ[i])^2 for i in 1:N)
                H!(core, u, (Nx=nx, Nt=nt, dx=dx, u0=x_1_b[:, 1, :], n_samples=n_samples, backend=backend))  
                nlp = ExaModel(core)
                result = madnlp(nlp, linear_solver=MadNLPGPU.CUDSSSolver)
                x_exa_vec = solution(result, u)
                x_0 = reshape(Float32.(x_exa_vec), nx, nt, 1, n_samples)
            else
                # CPU path: pull to CPU, use tuple embedding
                x_1_cpu = Array(x_1)                                     # (nx, nt, 1, n_samples)
                x_1_flat = x_1_cpu[:, :, 1, :]                          # (nx, nt, n_samples)
                u0_batch = x_1_flat[:, 1, :]                            # (nx, n_samples)
                x1_data = [(k, vec(x_1_flat)[k]) for k in 1:N]
                core = ExaCore(backend=backend)
                u = variable(core, 1:N, start = vec(x_1_flat))
                objective(core, (u[d[1]] - d[2])^2 for d in x1_data)
                p = (Nx=nx, Nt=nt, dx=dx, u0=u0_batch, n_samples=n_samples, backend=backend)
                H!(core, u, p)
                nlp = ExaModel(core)
                result = madnlp(nlp, print_level=MadNLP.ERROR)
                x_exa_vec = solution(result, u)
                # MadNLP returns Float64; cast back to Float32 before moving to device
                x_0 = reshape(Float32.(Array(x_exa_vec)), nx, nt, 1, n_samples) |> device
            end
        end
        ##############

        # Step 3: Interpolate between x_0 and x_1 (corrected) at time t+dt
        x = x_0 .+ (x_1 .- x_0) .* τ_next
    end

    return x
end
