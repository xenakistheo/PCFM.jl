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

#S

using PCFM
using Lux
#using Reactant
using JLD2, Functors
# using Plots
using cuDNN
using CUDA
using KernelAbstractions
using ExaModels, MadNLP, MadNLPGPU
using JuMP
using Ipopt
using BenchmarkTools

# include(joinpath(@__DIR__, "..", "optimisation", "plotUtils.jl"))


backend = CUDABackend()
backend isa GPU

dev_gpu = cu
dev_cpu = cpu_device

device = dev_gpu

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
    device = dev_gpu
)
println("  Model created successfully")

# 3. Load checkpoint

println("\n[3/5] Loading checkpoint from: $weight_file")
saved = JLD2.load(weight_file)
# device = ffm.config[:device]
device = cu
ps = saved["parameters"] |> device
st = saved["states"] |> device
losses = Float32[]
# compiled_funcs = PCFM.compile_functions(ffm, batch_size)
println("  Loaded trained parameters and states")


# Re-init Lux states for inference and move ps/st to device
# device = ffm.config[:device]
_, st = Lux.setup(Random.default_rng(), ffm.model)
ps = ps |> device
st = st |> device

# ---------------------------------------------------------------------------
# 5. Generate samples
# ---------------------------------------------------------------------------
println("\n[5/5] Generating samples...")
n_samples = 32
# sample_compiled_funcs = (n_samples == batch_size) ? compiled_funcs : PCFM.compile_functions(ffm, n_samples)
tstate_inf = (parameters = ps, states = st)

@show ffm.config[:device]


########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
function sample_pcfm(ffm::FFM, tstate, n_samples, n_steps, H!;
        constraint_parameters = nothing,
        domain = (x_start=0.0f0, x_end=2f0π, t_start=0.0f0, t_end=1.0f0),                                                                                        
        IC_func = x -> sin(x + π/4), 
        backend = CPU(),
        mode = "exa",
        optimizer = MadNLP.Optimizer,
        use_compiled = true,
        compiled_funcs = nothing,
        verbose = true,
        initial_vals=nothing)

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

    x_grid = range(domain.x_start, domain.x_end, length=nx)
    u_0_ic_vals = Float32.(IC_func.(x_grid))                          # (nx,)
    u_0_ic_mat  = repeat(reshape(u_0_ic_vals, nx, 1), 1, n_samples)  # (nx, n_samples)
    u_0_ic_mat = KernelAbstractions.adapt(backend, u_0_ic_mat)

    if initial_vals !== nothing
        @assert size(initial_vals) == (nx, nt, 1, n_samples)
        x_0 = initial_vals |> device
    else
        # Start from Gaussian noise
        x_0 = randn(Float32, nx, nt, 1, n_samples) |> device
    end 

    x = copy(x_0)

    dt = 1.0f0 / n_steps
    dx = x_grid[2] - x_grid[1]

    grid_points = (nx)
    grid_spacing = (dx)
    t_vec = fill(0f0, n_samples) |> device

    #Used to be in loop 
    N = nx * nt * n_samples
    
    # Define Optimization problem 
    if mode == "exa"
        x1_param = KernelAbstractions.adapt(backend, zeros(Float32, N))       # mutable, lives on GPU  
        core = ExaCore(backend=backend)                                                                                                                                                                                                     
        θ = parameter(core, x1_param)              # θ references x1_param by address                                                                                                                                                       
        u = variable(core, 1:N; start = x1_param)  
        objective(core, (u[i] - θ[i])^2 for i in 1:N)
        H!(core, u, u_0_ic_mat, nt, n_samples, grid_points, grid_spacing, dt, constraint_parameters; backend=backend)                                                                                                                   
        nlp = ExaModel(core)                                                                                                                                                                                                                
        
        if backend isa GPU
            solver = MadNLP.MadNLPSolver(nlp; linear_solver=MadNLPGPU.CUDSSSolver, print_level=MadNLP.ERROR) 
        else 
            solver = MadNLP.MadNLPSolver(nlp; print_level=MadNLP.ERROR)
        end 
    end 

    # Euler integration from t=0 to t=1
    for step in 0:(n_steps - 1)
        if verbose && step % 10 == 0
            println("PCFM step: $step/$n_steps")
        end

        τ = step * dt
        τ_next = τ + dt
        fill!(t_vec, τ)

        # Prepare input with embeddings
        x_input = prepare_input_fn(x, t_vec, (nx,), nt, n_samples, emb_channels) 

        # Predict velocity field
        v, st = model_fn(x_input, ps, st)

        # Step 1: Extrapolate to t=1
        x_1 = x .+ v .* (1.0f0 - τ) 

        # Step 2: Apply constraint - fix initial condition
        ##############
       
        # ExaModel version 

        if backend isa GPU
            copyto!(x1_param, reshape(x_1, N))     # no new allocations, no PCIe transfer
            result = MadNLP.solve!(solver)                 # reuses factorization structure
            x_0 = reshape(Float32.(solution(result, u)), nx, nt, 1, n_samples) |> device
        else
            # CPU path: pull to CPU, use tuple embedding
            copyto!(x1_param, reshape(x_1, N))
            result = MadNLP.solve!(solver)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
            x_0 = reshape(Float32.(solution(result, u)), nx, nt, 1, n_samples) |> device
        end
        ##############

        # Step 3: Interpolate between x_0 and x_1 (corrected) at time t+dt
        @. x = x_0 + (x_1 - x_0) * τ_next 
    end

    return Array(x)
end
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
function sample_pcfm_old(ffm::FFM, tstate, n_samples, n_steps, H!;
        constraint_parameters = nothing,
        domain = (x_start=0.0f0, x_end=2f0π, t_start=0.0f0, t_end=1.0f0),                                                                                        
        IC_func = x -> sin(x + π/4), 
        backend = CPU(),
        mode = "exa",
        optimizer = MadNLP.Optimizer,
        use_compiled = true,
        compiled_funcs = nothing,
        verbose = true,
        initial_vals=nothing)

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

    x_grid = range(domain.x_start, domain.x_end, length=nx)
    u_0_ic_vals = Float32.(IC_func.(x_grid))                          # (nx,)
    u_0_ic_mat  = repeat(reshape(u_0_ic_vals, nx, 1), 1, n_samples)  # (nx, n_samples)
    u_0_ic_mat = KernelAbstractions.adapt(backend, u_0_ic_mat)


    if initial_vals !== nothing
        @assert size(initial_vals) == (nx, nt, 1, n_samples)
        x_0 = initial_vals |> device
    else
        # Start from Gaussian noise
        x_0 = randn(Float32, nx, nt, 1, n_samples) |> device
    end 
    x = copy(x_0)

    dt = 1.0f0 / n_steps
    dx = x_grid[2] - x_grid[1]

    grid_points = (nx)
    grid_spacing = (dx)
    t_vec = fill(0f0, n_samples) |> device

    # Euler integration from t=0 to t=1
    for step in 0:(n_steps - 1)
        if verbose && step % 10 == 0
            println("PCFM step: $step/$n_steps")
        end

        τ = step * dt
        τ_next = τ + dt
        fill!(t_vec, τ)

        # Prepare input with embeddings
        x_input = prepare_input_fn(x, t_vec, (nx,), nt, n_samples, emb_channels) 

        # Predict velocity field
        v, st = model_fn(x_input, ps, st)

        # Step 1: Extrapolate to t=1
        x_1 = x .+ v .* (1.0f0 - τ) 

        # Step 2: Apply constraint - fix initial condition
        ##############
        if mode == "jump"
            # JuMP version — batched over all samples at once
            x_1_cpu = Array(x_1)  # (nx, nt, 1, n_samples)
            model = Model(optimizer)
            set_silent(model)
            @variable(model, u[1:nx, 1:nt, 1:n_samples])
            @objective(model, Min, sum((u[i,j,s] - x_1_cpu[i,j,1,s])^2 for i in 1:nx, j in 1:nt, s in 1:n_samples))
            H!(model, u, x_1_cpu, nt, n_samples, grid_points, grid_spacing, dt, constraint_parameters)
            optimize!(model)
            x_0 = reshape(Float32.(value.(u)), nx, nt, 1, n_samples) |> device
        else
            # ExaModel version 
            N = nx * nt * n_samples

            if backend isa GPU
                x1_vec = reshape(x_1, N)
                core = ExaCore(backend=backend)
                θ = parameter(core, x1_vec)
                u = variable(core, 1:N; start = x1_vec)
                objective(core, (u[i] - θ[i])^2 for i in 1:N)
                H!(core, u, u_0_ic_mat, nt, n_samples, grid_points, grid_spacing, dt, constraint_parameters; backend=backend)
                nlp = ExaModel(core)
                result = madnlp(nlp, linear_solver=MadNLPGPU.CUDSSSolver, print_level = MadNLP.ERROR)
                x_exa_vec = solution(result, u)
                x_0 = reshape(Float32.(x_exa_vec), nx, nt, 1, n_samples) |> device
            else
                # CPU path: pull to CPU, use tuple embedding
                x_flat_vec = vec(Array(x_1))   # (N,) on CPU; singleton dim collapses naturally                                                                                                                                                     
                core = ExaCore(backend=backend)                                                                                                                                                                                                     
                u = variable(core, 1:N, start = x_flat_vec)                                                                                                                                                                                         
                objective(core, (u[i] - v)^2 for (i, v) in enumerate(x_flat_vec))                                                                                                                                                                   
                H!(core, u, u_0_ic_mat, nt, n_samples, grid_points, grid_spacing, dt, constraint_parameters; backend=backend)                                                                                                                       
                nlp = ExaModel(core)
                result = madnlp(nlp, print_level=MadNLP.ERROR)                                                                                                                                                                                      
                x_0 = reshape(Float32.(solution(result, u)), nx, nt, 1, n_samples) |> device
            end
        end
        ##############

        # Step 3: Interpolate between x_0 and x_1 (corrected) at time t+dt
        @. x = x_0 + (x_1 - x_0) * τ_next 
    end

    return Array(x)
end
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
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

########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################


@show backend
##############
# ExaModels, MadNLP, GPU
# @btime sample_pcfm($ffm, (parameters = $ps, states = $st),
#                    $n_samples, 100, heat_constraints!;
#                    backend=backend,
#                    verbose = true,
#                    mode="exa");

starting_noise = randn(Float32, nx, nt, 1, n_samples) 

samples_exa_gpu = sample_pcfm(ffm, (parameters = ps, states = st),
                   n_samples, 100, heat_constraints!;
                   backend=backend,
                   verbose = true,
                   mode="exa", 
                   initial_vals=starting_noise);

samples_exa_gpu_old = sample_pcfm_old(ffm, (parameters = ps, states = st),
                   n_samples, 100, heat_constraints!;
                   backend=backend,
                   verbose = true,
                   mode="exa",
                   initial_vals=starting_noise);

#80.824 s (11250911 allocations: 14.42 GiB)

# # ExaModels, MadNLP, CPU
# @btime sample_pcfm($ffm, (parameters = $ps, states = $st),
#                    $n_samples, 100, heat_constraints!;
#                    backend=CPU(),
#                    verbose = true,
#                    mode="exa");
# 37.737 s (1544423 allocations: 20.34 GiB)


samples_exa_cpu = sample_pcfm(ffm, (parameters = ps, states = st),
                   n_samples, 100, heat_constraints!;
                   backend=CPU(),
                   verbose = true,
                   mode="exa", 
                   initial_vals=starting_noise);

samples_exa_cpu_old = sample_pcfm_old(ffm, (parameters = ps, states = st),
                   n_samples, 100, heat_constraints!;
                   backend=CPU(),
                   verbose = true,
                   mode="exa", 
                   initial_vals=starting_noise);

# #JuMP, MadNLP
# @btime sample_pcfm($ffm, (parameters = $ps, states = $st),
#                    $n_samples, 100, heat_constraints!;
#                    backend=CPU(),
#                    verbose = true,
#                    mode="jump",
#                    optimizer=MadNLP.Optimizer);

samples_jump_madnlp = sample_pcfm_old(ffm, (parameters = ps, states = st),
                   n_samples, 100, heat_constraints!;
                   backend=CPU(),
                   verbose = true,
                   mode="jump",
                   optimizer=MadNLP.Optimizer);


# #JuMP, Ipopt
# @btime sample_pcfm($ffm, (parameters = $ps, states = $st),
#                    $n_samples, 100, heat_constraints!;
#                    backend=CPU(),
#                    verbose = true,
#                    mode="jump",
#                    optimizer=Ipopt.Optimizer);


samples_ffm = sample_ffm(ffm, (parameters = ps, states = st), n_samples, 100; 
    verbose = true,
    initial_vals=starting_noise)

##################
# Plot solutions to verify correctness 


######################

new_samples = Array(samples_exa_gpu)
old_samples = Array(samples_exa_gpu_old)
ffm_samples = Array(samples_ffm)

using CairoMakie
CairoMakie.activate!()


X = x_grid
T = range(t_range[1], t_range[2]; length = nt)

function plot_sample(k, u1, u2, u3, title="")
    f = Figure(size = (1800, 600))

    ax1 = Axis(f[1, 1], 
                title = "New PCFM",
                xlabel = "Time", 
                ylabel = "X")

    ax2 = Axis(f[1, 2], 
                title = "Old PCFM",
                xlabel = "Time", 
                ylabel = "X")

    ax3 = Axis(f[1, 3], 
                title = "FFM",
                xlabel = "Time", 
                ylabel = "X")


    heatmap!(ax1, T, X, u1[:,:,1,k]', colormap = :viridis)
    heatmap!(ax2, T, X, u2[:,:,1,k]', colormap = :viridis)
    heatmap!(ax3, T, X, u3[:,:,1,k]', colormap = :viridis)


    Colorbar(f[1, 4], label = "Amplitude")
    Label(f[0, :], title)

    return f
end 

# f = Figure(size = (1800, 600))
# ax[1] = Axis(f[1, 1], 
#                 title = "New PCFM",
#                 xlabel = "Time", 
#                 ylabel = "X")


plot_sample(1, new_samples, old_samples, samples_jump_madnlp)
plot_sample(2, new_samples, old_samples, ffm_samples)
plot_sample(3, new_samples, old_samples, ffm_samples)
plot_sample(4, new_samples, old_samples, ffm_samples)
plot_sample(5, new_samples, old_samples, ffm_samples)

#CPU seems equal to GPU. But old PCFM and new PCFM do not seem equal. 
# Old version varies across samples, while new version seems rather fixed.
# None of them look like the FFM. 


function mass_constraint(u, params)
    Nx, Nt = params
    return [sum((u[i, j] - u[i,1]) for i in 1:(Nx-1)) for j in 1:Nt]
end 

mass_constraint(new_samples[:,:,1,1], (nx, nt))

function plot_constraint_violation(k, u1, u2, u3, H; title="", constraint_params=nothing)
    f = Figure(size = (1800, 600))

    ax1 = Axis(f[1, 1], 
                title = "New PCFM",
                xlabel = "Time", 
                ylabel = "Violation")

    ax2 = Axis(f[1, 2], 
                title = "Old PCFM",
                xlabel = "Violation", 
                ylabel = "X")

    ax3 = Axis(f[1, 3], 
                title = "FFM",
                xlabel = "Time", 
                ylabel = "Violation")


    lines!(ax1, H(u1[:,:,1,k], constraint_params))
    lines!(ax2, H(u2[:,:,1,k], constraint_params))
    lines!(ax3, H(u3[:,:,1,k], constraint_params))


    Label(f[0, :], title)

    return f
end 

plot_constraint_violation(1, new_samples, old_samples, ffm_samples, mass_constraint; constraint_params=(nx, nt, dx, dt))


u1 = randn(nx, nt, n_samples)
L = 2π
X = collect(range(0, L; length=nx))
T = collect(range(0, 1; length=nt))
core = ExaCore(Float64; backend=backend)
N = nx * nt * n_samples
u = variable(core, 1:N, start = vec(u1))
u1_data = KernelAbstractions.adapt(backend, [(k, vec(u1)[k]) for k in 1:N])
objective(core, (u[d[1]] - d[2])^2 for d in u1_data)
heat_constraints!(core, u, u1[:, 1, :], nt, n_samples, [nx], [dx], dt; backend=backend)
nlp = ExaModel(core)
result = madnlp(nlp)
u_exa_vec = solution(result, u)
u_exa = Array(reshape(u_exa_vec, nx, nt, 1, n_samples))

plot_constraint_violation(1, new_samples, old_samples, u_exa, mass_constraint; constraint_params=(nx, nt, dx, dt))

