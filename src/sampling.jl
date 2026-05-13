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
        # Start from Gaussian noise
        x = randn(Float32, nx, nt, 1, n_samples) |> device
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


function reactant_residuals(
    H!,
    z,
    u0,
    nt,
    n_samples,
    grid_points,
    grid_spacing,
    dt,
    params;
    backend = CPU(),
)
    return H!(
        ReactantResidualModel(),
        z,
        u0,
        nt,
        n_samples,
        grid_points,
        grid_spacing,
        dt,
        params;
        backend = backend,
    )
end


function reactant_Jv(
    H!,
    z,
    v,
    u0,
    nt,
    n_samples,
    grid_points,
    grid_spacing,
    dt,
    params,
    backend,
)
    h_z = z_local -> reactant_residuals(
        H!,
        z_local,
        u0,
        nt,
        n_samples,
        grid_points,
        grid_spacing,
        dt,
        params;
        backend = backend,
    )

    tangent_tuple = Enzyme.autodiff(
        Enzyme.Forward,
        h_z,
        Enzyme.Duplicated,
        Enzyme.Duplicated(z, v),
    )

    return only(tangent_tuple)
end


function reactant_Jtλ(
    H!,
    z,
    λ,
    u0,
    nt,
    n_samples,
    grid_points,
    grid_spacing,
    dt,
    params,
    backend,
)
    scalarized = z_local -> begin
        r = reactant_residuals(
            H!,
            z_local,
            u0,
            nt,
            n_samples,
            grid_points,
            grid_spacing,
            dt,
            params;
            backend = backend,
        )

        return sum(r .* λ)
    end

    grad_tuple = Enzyme.gradient(
        Enzyme.Reverse,
        scalarized,
        z,
    )

    return only(grad_tuple)
end


function reactant_cg(A, b, maxiter, reg_eps)
    x = zero(b)
    r = b .- A(x)
    p = r

    rsold = sum(r .* r, dims = 1)

    for _ in 1:maxiter
        Ap = A(p)

        denom = sum(p .* Ap, dims = 1)
        denom = denom .+ reg_eps

        α = rsold ./ denom

        x = x .+ p .* α
        r = r .- Ap .* α

        rsnew = sum(r .* r, dims = 1)

        β = rsnew ./ (rsold .+ reg_eps)

        p = r .+ p .* β
        rsold = rsnew
    end

    return x
end


function reactant_matrix_free_sqp_step(
    H!,
    z,
    xhat,
    u0,
    nt,
    n_samples,
    grid_points,
    grid_spacing,
    dt,
    params,
    backend,
    reg,
    cg_iters,
)
    r = reactant_residuals(
        H!,
        z,
        u0,
        nt,
        n_samples,
        grid_points,
        grid_spacing,
        dt,
        params;
        backend = backend,
    )

    diff = xhat .- z

    Jdiff = reactant_Jv(
        H!,
        z,
        diff,
        u0,
        nt,
        n_samples,
        grid_points,
        grid_spacing,
        dt,
        params,
        backend,
    )

    rhs = Jdiff .+ r

    A = λ -> begin
        Jtλ = reactant_Jtλ(
            H!,
            z,
            λ,
            u0,
            nt,
            n_samples,
            grid_points,
            grid_spacing,
            dt,
            params,
            backend,
        )

        JJtλ = reactant_Jv(
            H!,
            z,
            Jtλ,
            u0,
            nt,
            n_samples,
            grid_points,
            grid_spacing,
            dt,
            params,
            backend,
        )

        return JJtλ .+ reg .* λ
    end

    λ = reactant_cg(A, rhs, cg_iters, eps(eltype(rhs)))

    Jtλ = reactant_Jtλ(
        H!,
        z,
        λ,
        u0,
        nt,
        n_samples,
        grid_points,
        grid_spacing,
        dt,
        params,
        backend,
    )

    dz = diff .- Jtλ

    return z .+ dz
end


function reactant_matrix_free_project(
    H!,
    xhat,
    u0,
    nt,
    n_samples,
    grid_points,
    grid_spacing,
    dt,
    params,
    backend,
    maxiter,
    reg,
    cg_iters,
)
    z = copy(xhat)

    for _ in 1:maxiter
        z = reactant_matrix_free_sqp_step(
            H!,
            z,
            xhat,
            u0,
            nt,
            n_samples,
            grid_points,
            grid_spacing,
            dt,
            params,
            backend,
            reg,
            cg_iters,
        )
    end

    return z
end


function build_reactant_matrix_free_projection(
    H!,
    state_size,
    ic_size,
    n_samples,
    nt,
    grid_points,
    grid_spacing,
    dt,
    params,
    backend;
    maxiter = 4,
    reg = 1f-6,
    cg_iters = 10,
)
    params = params === nothing ? (;) : params

    xhat_dummy = Reactant.to_rarray(randn(Float32, state_size, n_samples))
    u0_dummy = Reactant.to_rarray(randn(Float32, ic_size, n_samples))

    kernel = (xhat, u0) -> reactant_matrix_free_project(
        H!,
        xhat,
        u0,
        nt,
        n_samples,
        grid_points,
        grid_spacing,
        Float32(dt),
        params,
        backend,
        maxiter,
        Float32(reg),
        cg_iters,
    )

    println("Compiling matrix-free Reactant projection...")
    compiled = Reactant.@compile kernel(xhat_dummy, u0_dummy)
    println("Matrix-free Reactant projection compiled.")

    return compiled
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
function sample_pcfm(ffm::FFM, tstate, n_samples, n_steps, H!;
        constraint_parameters = nothing,
        domain = (x_start = 0.0f0, x_end = 2f0π, t_start = 0.0f0, t_end = 1.0f0),
        IC_func = x -> sin(x + π / 4),
        backend = CPU(),
        mode = "exa",
        optimizer = MadNLP.Optimizer,
        use_compiled = true,
        compiled_funcs = nothing,
        verbose = true,
        initial_vals = nothing,
        reactant_maxiter = 8,
        reactant_reg = 1f-7,
        reactant_cg_iters = 30)

    nx = ffm.config[:nx]
    nt = ffm.config[:nt]
    emb_channels = ffm.config[:emb_channels]
    device = ffm.config[:device]

    println("\n------------------------")
    println("------Sampling PCFM------")
    println("Modelling: $mode")
    println("Optimizer: ", string(optimizer))
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
        model_fn = compiled_funcs.model
        prepare_input_fn = compiled_funcs.prepare_input
    else
        model_fn = ffm.model
        prepare_input_fn = prepare_input
    end

    x_grid = range(domain.x_start, domain.x_end, length = nx)
    u_0_ic_vals = Float32.(IC_func.(x_grid))

    u_0_ic_mat = KernelAbstractions.adapt(
        backend,
        repeat(reshape(u_0_ic_vals, nx, 1), 1, n_samples),
    )

    if initial_vals !== nothing
        @assert size(initial_vals) == (nx, nt, 1, n_samples)
        x_0 = initial_vals |> device
    else
        x_0 = randn(Float32, nx, nt, 1, n_samples) |> device
    end

    x = copy(x_0)

    dt = 1.0f0 / n_steps
    dx = x_grid[2] - x_grid[1]

    grid_points = (nx,)
    grid_spacing = (dx,)

    spatial_dims = grid_points
    state_size = prod(spatial_dims) * nt
    ic_size = prod(spatial_dims)
    N = state_size * n_samples

    t_vec = fill(0f0, n_samples) |> device

    if mode == "jump"
        u_0_ic_mat_jump = reshape(u_0_ic_mat, nx, 1, 1, n_samples)

    elseif mode == "exa"
        x1_param = KernelAbstractions.adapt(backend, zeros(Float32, N))

        core = ExaCore(backend = backend)
        θ = parameter(core, x1_param)
        u = variable(core, 1:N; start = x1_param)

        objective(core, (u[i] - θ[i])^2 for i in 1:N)

        H!(
            core,
            u,
            u_0_ic_mat,
            nt,
            n_samples,
            grid_points,
            grid_spacing,
            dt,
            constraint_parameters;
            backend = backend,
        )

        nlp = ExaModel(core)

        if backend isa GPU
            solver = MadNLP.MadNLPSolver(
                nlp;
                linear_solver = MadNLPGPU.CUDSSSolver,
                print_level = MadNLP.ERROR,
            )
        else
            solver = MadNLP.MadNLPSolver(
                nlp;
                print_level = MadNLP.ERROR,
            )
        end

    elseif mode == "reactant"
        reactant_project = build_reactant_matrix_free_projection(
            H!,
            state_size,
            ic_size,
            n_samples,
            nt,
            grid_points,
            grid_spacing,
            dt,
            constraint_parameters,
            backend;
            maxiter = reactant_maxiter,
            reg = reactant_reg,
            cg_iters = reactant_cg_iters,
        )
    
        u0_reactant = Reactant.to_rarray(
            reshape(Float32.(Array(u_0_ic_mat)), ic_size, n_samples)
        )

    else
        error("Unknown mode: $mode. Expected \"jump\", \"exa\", or \"reactant\".")
    end

    for step in 0:(n_steps - 1)
        verbose && step % 10 == 0 && println("PCFM step: $step/$n_steps")

        τ = step * dt
        τ_next = τ + dt

        fill!(t_vec, Float32(τ))

        x_input = prepare_input_fn(x, t_vec, spatial_dims, nt, n_samples, emb_channels)
        v, st = model_fn(x_input, ps, st)

        x_1 = x .+ v .* (1.0f0 - Float32(τ))

        if mode == "jump"
            x_1_cpu = Array(x_1)

            model = Model(optimizer)
            set_silent(model)

            @variable(model, u[1:nx, 1:nt, 1:n_samples])

            @objective(
                model,
                Min,
                sum(
                    (u[i, j, s] - x_1_cpu[i, j, 1, s])^2
                    for i in 1:nx, j in 1:nt, s in 1:n_samples
                ),
            )

            H!(
                model,
                u,
                u_0_ic_mat_jump,
                nt,
                n_samples,
                grid_points,
                grid_spacing,
                dt,
                constraint_parameters,
            )

            optimize!(model)

            x_1 = reshape(Float32.(value.(u)), nx, nt, 1, n_samples) |> device

        elseif mode == "exa"
            copyto!(nlp.θ, reshape(x_1, N))

            result = MadNLP.solve!(solver)

            x_1 = reshape(
                Float32.(solution(result, u)),
                nx,
                nt,
                1,
                n_samples,
            ) |> device

        elseif mode == "reactant"
            xhat_reactant = Reactant.to_rarray(
                reshape(Float32.(Array(x_1)), state_size, n_samples)
            )

            z_reactant = reactant_project(xhat_reactant, u0_reactant)

            x_1 = reshape(
                Array(z_reactant),
                spatial_dims...,
                nt,
                1,
                n_samples,
            ) |> device

        else
            error("Unknown mode: $mode.")
        end

        @. x = x_0 + (x_1 - x_0) * Float32(τ_next)
    end

    return Array(x)
end