"""
Example script for training and sampling from a Functional Flow Matching model
on the 1D heat (diffusion) equation.

This script demonstrates:
1. Creating an FFM model
2. Generating training data
3. Compiling functions with Reactant
4. Training the model
5. Generating unconstrained and physics-constrained samples
"""

using PCFM
using Reactant, Lux
using JLD2, Functors
using Plots
using CUDA
using KernelAbstractions


backend = CUDABackend()
backend isa GPU
dev_gpu = cu

# Set random seed
using Random
Random.seed!(1234)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
batch_size   = 32
nx           = 100          # Spatial resolution
nt           = 100          # Temporal resolution
emb_channels = 32
n_epochs     = 1000
force_retrain = false

# Checkpoint path
weight_file = joinpath(@__DIR__, "checkpoints", "ffm_heat_checkpoint.jld2")

# Data generation parameters
visc_range = (1.0f0, 5.0f0)
phi_range  = (0.0f0, Float32(π))
t_range    = (0.0f0, 1.0f0)

# Grid
x_grid = range(0.0f0, 2.0f0*Float32(π); length = nx)
dx     = Float32(x_grid[2] - x_grid[1])
dt     = 1.0f0 / (nt - 1)

# Initial condition: u(x, 0) = sin(x + π/4)
u0_ic = Float32.(sin.(x_grid .+ π/4))

# Constraint params (passed through to heat_constraints!)
constraint_params = (Nx=nx, Nt=nt, dx=dx)

# ---------------------------------------------------------------------------
println("=" ^ 60)
println("Heat Equation — Functional Flow Matching")
println("=" ^ 60)

# 1. Generate training data
println("\n[1/5] Generating training data...")
u_data = generate_diffusion_data(batch_size, nx, nt, visc_range, phi_range, t_range)
println("  Data shape: $(size(u_data))")

# 2. Create model
println("\n[2/5] Creating FFM model...")
ffm = FFM(
    nx = nx,
    nt = nt,
    emb_channels = emb_channels,
    hidden_channels = 64,
    proj_channels = 256,
    n_layers = 4,
    modes = (32, 32),
    device = reactant_device()
)
println("  Model created successfully")

# 3. Load checkpoint or train
if isfile(weight_file) && !force_retrain
    println("\n[3/5] Loading checkpoint from: $weight_file")
    saved = JLD2.load(weight_file)
    device = ffm.config[:device]
    ps = saved["parameters"] |> device
    st = saved["states"] |> device
    losses = Float32[]
    compiled_funcs = PCFM.compile_functions(ffm, batch_size)
    println("  Loaded trained parameters and states")
else
    println("\n[3/5] Compiling functions with Reactant...")
    compiled_funcs = PCFM.compile_functions(ffm, batch_size)

    println("\n[4/5] Training model for $n_epochs epochs...")
    losses, tstate = train_ffm!(ffm, u_data; compiled_funcs, epochs = n_epochs, verbose = true)
    println("\nFinal loss: $(losses[end])")

    ps = fmap(x -> x isa AbstractArray ? Array(x) : x, tstate.parameters)
    st = fmap(x -> x isa AbstractArray ? Array(x) : x, tstate.states)

    println("Saving checkpoint to: $weight_file")
    mkpath(dirname(weight_file))
    JLD2.save(weight_file, "parameters", ps, "states", st, "config", ffm.config)
end

# Re-init Lux states for inference and move ps/st to device
device = ffm.config[:device]
_, st = Lux.setup(Random.default_rng(), ffm.model)
ps = ps |> device
st = st |> device

# ---------------------------------------------------------------------------
# 5. Generate samples
# ---------------------------------------------------------------------------
println("\n[5/5] Generating samples...")
n_samples = 32
sample_compiled_funcs = (n_samples == batch_size) ? compiled_funcs : PCFM.compile_functions(ffm, n_samples)
tstate_inf = (parameters = ps, states = st)



########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################


function sample_pcfm(ffm::FFM, tstate, n_samples, n_steps, H!, params;
        backend = CPU(),
        mode = "exa",
        use_compiled = true,
        compiled_funcs = nothing,
        verbose = true)

    nx = ffm.config[:nx]
    nt = ffm.config[:nt]
    emb_channels = ffm.config[:emb_channels]
    device = ffm.config[:device]

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
    x_0 = randn(Float32, nx, nt, 1, n_samples) |> device
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
        t_vec = fill(τ, n_samples) |> device

        # Prepare input with embeddings
        x_input = prepare_input_fn(x, t_vec, nx, nt, n_samples, emb_channels)

        # Predict velocity field
        v, st = model_fn(x_input, ps, st)

        # Step 1: Extrapolate to t=1
        x_1 = x .+ v .* (1.0f0 - τ)

        # Step 2: Apply constraint - fix initial condition
        @. x_1[:, 1:1, :, :] = u_0_ic
        ##############
        if mode == "jump"
            # JuMP version — batched over all samples at once
            x_1_cpu = Array(x_1)  # (nx, nt, 1, n_samples)
            model = Model(MadNLP.Optimizer)
            set_silent(model)
            @variable(model, u[1:nx, 1:nt, 1:n_samples])
            @objective(model, Min, sum((u[i,j,s] - x_1_cpu[i,j,1,s])^2 for i in 1:nx, j in 1:nt, s in 1:n_samples))
            @constraint(model, [j in 1:nt, s in 1:n_samples], dx * sum(u[i,j,s] for i in 1:(nx-1)) == 0.0)
            optimize!(model)
            x_0 = reshape(Float32.(value.(u)), nx, nt, 1, n_samples) |> device
        else
            # ExaModel version — solve projection for all samples at once
            N = nx * nt * n_samples

            if backend isa GPU
                # GPU path: keep data on device, use array iteration (no scalar indexing)
                @show typeof(x_1)
                @show backend(x_1)
                x_1_flat = x_1[:, :, 1, :]                              # (nx, nt, n_samples) on device
                u0_batch = x_1_flat[:, 1, :]                            # (nx, n_samples) on device
                indices = KernelAbstractions.adapt(backend, collect(1:N))
                values  = KernelAbstractions.adapt(backend, vec(x_1_flat))
                core = ExaCore(backend=backend)
                u = variable(core, 1:N, start = values)
                objective(core, (u[i] - values[i])^2 for i in indices)
                p = (Nx=nx, Nt=nt, dx=dx, u0=u0_batch, n_samples=n_samples, backend=backend)
                H!(core, u, p)
                nlp = ExaModel(core)
                result = madnlp(nlp, linear_solver=MadNLPGPU.LapackCUDASolver, print_level=MadNLP.ERROR)
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



function prepare_input(x_t, t, nx, nt, n_samples, emb_dim; max_positions = 2000)
    u_channel = x_t

    # Time embedding (sinusoidal)
    timesteps = t .* Float32(max_positions)
    half_dim = emb_dim ÷ 2

    emb_scale = Float32(log(max_positions)) / Float32(half_dim - 1)
    emb_base = exp.(Float32.(-collect(0:(half_dim - 1)) .* emb_scale))
    t_emb = timesteps * emb_base'
    t_emb = hcat(sin.(t_emb), cos.(t_emb))

    # Reshape and broadcast to spatial dimensions
    t_emb = permutedims(t_emb, (2, 1))
    t_emb = reshape(t_emb, 1, 1, emb_dim, n_samples)
    t_emb = repeat(t_emb, nx, nt, 1, 1)

    # Position embeddings (normalized coordinates)
    pos_x = range(0.0f0, 1.0f0, length = nx)
    pos_t = range(0.0f0, 1.0f0, length = nt)
    pos_x_grid = repeat(reshape(collect(pos_x), nx, 1, 1, 1), 1, nt, 1, n_samples)
    pos_t_grid = repeat(reshape(collect(pos_t), 1, nt, 1, 1), nx, 1, 1, n_samples)

    # Concatenate all channels
    x_input = cat(u_channel, t_emb, pos_x_grid, pos_t_grid; dims = 3)

    return x_input
end


########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################

# @time samples = sample_pcfm(ffm, (parameters = ps, states = st), n_samples, 100;
#     compiled_funcs = sample_compiled_funcs, verbose = true); #RUNS, 43.923413 seconds (1.70 M allocations: 491.470 MiB, 1.66% gc time, 0.12% compilation time)
# @time samples = sample_pcfm(ffm, (parameters = ps, states = st), n_samples, 100; verbose = true); #FAILS, needs compiled funcs. 

# @time samples_exa = sample_pcfm(ffm, (parameters = ps, states = st), n_samples, 100, heat_constraints!, nothing; 
#     backend=backend, verbose = true); #  

@time samples_exa = sample_pcfm(ffm, (parameters = ps, states = st), n_samples, 100, heat_constraints!, nothing; 
    backend=backend, compiled_funcs = sample_compiled_funcs, verbose = true); #FAILS 

@time samples_jump = sample_pcfm(ffm, (parameters = ps, states = st), n_samples, 100, heat_constraints!, nothing; 
    backend=CPU(), compiled_funcs = sample_compiled_funcs, verbose = true, mode="jump"); # RUNS, 212.163878 seconds (1.70 G allocations: 118.770 GiB, 31.97% gc time, 6.78% compilation times

# println("\n" * "=" ^ 60)
# println("Training Complete!")
# println("=" ^ 60)

# # Visualize results
# println("\nPlotting results...")

# # Plot training curve
# if !isempty(losses)
#     p1 = plot(1:length(losses), losses,
#         yscale = :log10,
#         xlabel = "Epoch",
#         ylabel = "Loss (log scale)",
#         title = "Training Loss",
#         legend = false,
#         linewidth = 2)
#     display(p1)
# end

# # Plot samples
# arr_data = Array(u_data)
# arr_samples = Array(samples)

# p_data = [Plots.heatmap(arr_data[:, :, 1, i],
#               title = "Training Data $i",
#               xlabel = "Time",
#               ylabel = "Space",
#               c = :viridis)
#           for i in 1:min(2, batch_size)]

# p_samples = [Plots.heatmap(arr_samples[:, :, 1, i],
#                  title = "Generated Sample $i",
#                  xlabel = "Time",
#                  ylabel = "Space",
#                  c = :viridis)
#              for i in 1:2]

# # Combine plots
# p2 = Plots.plot(p_data..., layout = (1, length(p_data)), size = (800, 300))
# p3 = Plots.plot(p_samples..., layout = (1, length(p_samples)), size = (800, 300))

# display(p2)
# display(p3)

# println("\nDone! Check the plots above.")
