"""
Example script for sampling from a Functional Flow Matching model
on the 1D heat (diffusion) equation.

Note: Script does not use Reactant
"""

using PCFM

using ExaModels, MadNLP, MadNLPGPU
# using Plots
using Lux
using CUDA
using cuDNN
using KernelAbstractions
using JLD2, Functors
using JuMP
using Ipopt
using BenchmarkTools
#using Reactant


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
    u_0_ic_mat  = KernelAbstractions.adapt(backend, repeat(reshape(u_0_ic_vals, nx, 1), 1, n_samples))  # (nx, n_samples)

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
    if mode == "jump"
        u_0_ic_mat = reshape(u_0_ic_mat, nx, 1, 1, n_samples)

    else #mode == "exa"
        x1_param = KernelAbstractions.adapt(backend, zeros(Float32, N))       # mutable, lives on GPU  
        core = ExaCore(backend=backend)                                                                                                                                                                                                     
        θ = parameter(core, x1_param)              # θ references x1_param by address                                                                                                                                                       
        u = variable(core, 1:N; start = x1_param)  
        objective(core, (u[i] - θ[i])^2 for i in 1:N)
        H!(core, u, u_0_ic_mat, nt, n_samples, grid_points, grid_spacing, dt, constraint_parameters; backend=backend)                                                                                                                   
        nlp = ExaModel(core)                                                                                                                                                                                                                
        
        if backend isa GPU
            solver = MadNLP.MadNLPSolver(nlp; linear_solver=MadNLPGPU.CUDSSSolver, print_level=MadNLP.ERROR, tol=1e-8)
        else
            solver = MadNLP.MadNLPSolver(nlp; print_level=MadNLP.ERROR, tol = 1e-6)
        end
        println("get_tolerance = ", MadNLP.get_tolerance(eltype(nlp.meta.x0), typeof(solver.opt.kkt_system)))
    end 

  

    # Euler integration from t=0 to t=1
    for step in 0:(n_steps - 1)
        if verbose && step % 5 == 0
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

        # Step 2: Apply constraints
        ##############
        # ExaModel version 
        if mode == "jump"
            x_1_cpu = Array(x_1)
            model = Model(optimizer)
            set_silent(model)
            @variable(model, u[1:nx, 1:nt, 1:n_samples])
            @objective(model, Min, sum((u[i,j,s] - x_1_cpu[i,j,1,s])^2 for i in 1:nx, j in 1:nt, s in 1:n_samples))
            H!(model, u, u_0_ic_mat, nt, n_samples, grid_points, grid_spacing, dt, constraint_parameters)
            optimize!(model)      

            x_1 = reshape(Float32.(value.(u)), nx, nt, 1, n_samples) |> device   
        else
            copyto!(nlp.θ, reshape(x_1, N))
            copyto!(nlp.meta.x0, reshape(x_1, N))
            # solver.opt.max_iter = 1
            MadNLP.set_status!(solver, MadNLP.INITIAL)
            result = MadNLP.solve!(solver)
            x_1 = reshape(Float32.(solution(result, u)), nx, nt, 1, n_samples) |> device
        end
        ##############

        # Step 3: Interpolate between x_0 and x_1 (corrected) at time t+dt
        @. x = x_0 + (x_1 - x_0) * τ_next 
    end

    # Final projection at t=1
    if mode == "jump"
        x_1_cpu = Array(x)
        model = Model(optimizer)
        set_silent(model)
        @variable(model, u[1:nx, 1:nt, 1:n_samples])
        @objective(model, Min, sum((u[i,j,s] - x_1_cpu[i,j,1,s])^2 for i in 1:nx, j in 1:nt, s in 1:n_samples))
        H!(model, u, u_0_ic_mat, nt, n_samples, grid_points, grid_spacing, dt, constraint_parameters)
        optimize!(model)
        x = reshape(Float32.(value.(u)), nx, nt, 1, n_samples) |> device
    else
        copyto!(nlp.θ, reshape(x, N))
        copyto!(nlp.meta.x0, reshape(x, N))
        # solver.opt.max_iter = 3
        MadNLP.set_status!(solver, MadNLP.INITIAL)
        result = MadNLP.solve!(solver)
        x = reshape(Float32.(solution(result, u)), nx, nt, 1, n_samples) |> device
    end

    println("Solver tolerance: ", solver.opt.tol)


    return Array(x)
end

@time samples_exa_gpu = sample_pcfm(ffm, (parameters = ps, states = st),
                n_samples, 100, heat_constraints!;
                backend=backend,
                verbose = false,
                mode="exa", 
                initial_vals=starting_noise);
typeof(samples_exa_gpu)

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
# batch_size   = 32
nx           = 100          # Spatial resolution
nt           = 100          # Temporal resolution
emb_channels = 32
n_epochs     = 1000
force_retrain = false

# Output path 
SAMPLES_PATH = length(ARGS) >= 2 ? ARGS[2] : "samples_heat.jld2"

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



########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################


n_samples = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 32
n_samples = 1

starting_noise = randn(Float32, nx, nt, 1, n_samples);



    # ExaModels, MadNLP, GPU
@info "ExaModels, MadNLP, GPU"
@time samples_exa_gpu = sample_pcfm(ffm, (parameters = ps, states = st),
                n_samples, 100, heat_constraints!;
                backend=backend,
                verbose = false,
                mode="exa", 
                initial_vals=starting_noise);


TYPE = Float64
samples_exa_gpu


batch_size   = 32
nx           = 100          # Spatial resolution
nt           = 100          # Temporal resolution
emb_channels = 32
t_range    = (0.0, 1.0)

x_grid = range(0.0, 2.0*(TYPE(π)); length = nx)
dx     = (x_grid[2] - x_grid[1])
dt     = 1.0f0 / (nt - 1f0)

# Initial condition: u(x, 0) = sin(x + π/4)
u0_ic = (sin.(x_grid .+ TYPE(π)/4))

X = x_grid
T = range(t_range[1], t_range[2]; length = nt)


function mass_constraint(u, params)
    Nx, Nt = params
    return [sum((u[i, j] - u[i,1]) for i in 1:(Nx-1)) for j in 1:Nt]
end

function ic_violation(u, params)
    nx, nt = params[1], params[2]
    return [sum(abs(u[j, i] - u[1, i]) for i in 1:nx) for j in 1:nt]
end

# Returns max absolute constraint residual (L∞ norm) across all constraints and samples,
# matching Ipopt/MadNLP's "Constraint violation" output.
# Handles both (nx, nt, n_samples) and (nx, nt, 1, n_samples) shapes.
function heat_constraint_violations(samples, u0_ic, nx, nt)
    u = ndims(samples) == 4 ? dropdims(samples, dims=3) : samples  # (nx, nt, n_samples)
    n_samples = size(u, 3)
    m0 = sum(u0_ic[1:nx-1])

    max_viol = 0.0

    for s in 1:n_samples
        # IC: u[i,1,s] == u0_ic[i] for i in 1:nx
        for i in 1:nx
            max_viol = max(max_viol, abs(u[i, 1, s] - u0_ic[i]))
        end

        # Mass: sum(u[1:nx-1,t,s]) == m0 for t in 2:nt
        for t in 2:nt
            max_viol = max(max_viol, abs(sum(u[1:nx-1, t, s]) - m0))
        end
    end

    return max_viol
end
typeof(viol)
viol = heat_constraint_violations(samples_exa_gpu, u0_ic, nx, nt) #4e-6
# viol = heat_constraint_violations(samples_ffm, u0_ic, nx, nt)
viol = heat_constraint_violations(u_analytic, u0_ic, nx, nt) #4e-6
##################

# Compute Analytic Solution 
X = x_grid
T = range(t_range[1], t_range[2]; length = nt)
u_exact = exp.(-3 .* T') .* sin.(X .+ π/4)   # (nx, nt), analytical solution ν=3
u_analytic = similar(samples_exa_gpu)
u_analytic[:,:, 1, 1] = u_exact
u_analytic




