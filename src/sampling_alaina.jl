# ================================================================ - IS THIS IN THE ORIGINAL CODE
#  sampling.jl — ODE loop, decoupled from projection backend
# ================================================================

"""
    sample_ffm(ffm, tstate, n_samples, n_steps; verbose)

Unconstrained Euler sampling (no projection).
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
    sample_pcfm(model, ps, st, nx, nt, emb_channels,
                n_samples, n_steps,
                solver::AbstractProjectionSolver,
                constraint_data;
                verbose = true)

Physics-constrained sampling.  The ODE loop knows **nothing** about
which solver or constraint is used — it just calls `solve_projection`.
"""
function sample_pcfm(model, ps, st, nx, nt, emb_channels,
    n_samples, n_steps,
    solver::AbstractProjectionSolver,
    constraint_data;
    verbose=true)

    spatial_size = nx isa Tuple ? nx : (nx,)

    x_0 = randn(Float32, spatial_size..., nt, 1, n_samples)
    x = copy(x_0)
    dt = 1.0f0 / n_steps
    # time_model = 0.0   
    # time_proj  = 0.0 

    for step in 0:(n_steps - 1)
        verbose && step % 10 == 0 && println("PCFM step: $step/$n_steps")

        τ = step * dt
        τ_next = τ + dt
        t_vec = fill(Float32(τ), n_samples)

        x_input = prepare_input(x, t_vec, spatial_size, nt, n_samples, emb_channels)
        # t0 = time()         
        v, st = model(x_input, ps, st)
        # time_model += time() - t0

        x_1 = x .+ v .* (1.0f0 - τ)
        # t0 = time()
        x_1 = solve_projection(solver, x_1, constraint_data)
        # time_proj += time() - t0 
        x = x_0 .+ (x_1 .- x_0) .* τ_next
    end

    # @info "Timing breakdown" time_model time_proj proj_perc=(time_proj/(time_model+time_proj))*100 model_perc=(time_model/(time_model+time_proj))*100   # ← report
    return x
end

function _unpack_tstate(tstate)
    if hasfield(typeof(tstate), :parameters)
        return tstate.parameters, tstate.states
    else
        return tstate[1], tstate[2]
    end
end