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
using ExaModels, MadNLP, MadNLPGPU
using JuMP
using Ipopt
using BenchmarkTools


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
weight_file = joinpath(@__DIR__, "..", "checkpoints", "ffm_heat_checkpoint.jld2")

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

########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################

########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################


# Runs 

# #The following benchmarks were recorded using the following compute 
# # salloc -p mit_normal_gpu --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=03:00:00

# #1st run 236.599683 seconds (493.11 M allocations: 41.098 GiB, 4.62% gc time, 1 lock conflict, 51.06% compilation time: <1% of which was recompilation)
# #2nd run  98.144399 seconds (11.48 M allocations: 15.121 GiB, 2.81% gc time)
# @time samples_exa = sample_pcfm(ffm, (parameters = ps, states = st), n_samples, 20, heat_constraints!; 
#     backend=backend, compiled_funcs = sample_compiled_funcs, verbose = true); 


# #1st run 101.670634 seconds (192.11 M allocations: 29.090 GiB, 22.53% gc time, 24.50% compilation time)
# #2nd run 83.031268 seconds (35.11 M allocations: 23.100 GiB, 28.98% gc time, 0.80% compilation time)
# @time samples_exa = sample_pcfm(ffm, (parameters = ps, states = st), n_samples, 20, heat_constraints!; 
#     backend=CPU(), compiled_funcs = sample_compiled_funcs, verbose = true);  

# # JuMP - Compiled Functions 
# #1st run 212.458600 seconds
# #2nd run 175.130955 seconds (1.62 G allocations: 111.857 GiB, 33.80% gc time)
# @time samples_jump = sample_pcfm(ffm, (parameters = ps, states = st), n_samples, 20, heat_constraints!; 
#     backend=CPU(), compiled_funcs = sample_compiled_funcs, verbose = true, mode="jump"); 

# @time samples_jump = sample_pcfm(ffm, (parameters = ps, states = st), n_samples, 20, heat_constraints!; 
#     backend=CPU(), compiled_funcs = sample_compiled_funcs, verbose = true, mode="jump", optimizer=Ipopt.Optimizer); 




##############
# ExaModels, MadNLP, GPU
@btime sample_pcfm($ffm, (parameters = $ps, states = $st),
                   $n_samples, 100, heat_constraints!;
                   backend=backend,
                   compiled_funcs = sample_compiled_funcs,
                   verbose = true,
                   mode="exa");

# ExaModels, MadNLP, CPU
@btime sample_pcfm($ffm, (parameters = $ps, states = $st),
                   $n_samples, 100, heat_constraints!;
                   backend=CPU(),
                   compiled_funcs = sample_compiled_funcs,
                   verbose = true,
                   mode="exa");


#JuMP, MadNLP
@btime sample_pcfm($ffm, (parameters = $ps, states = $st),
                   $n_samples, 100, heat_constraints!;
                   backend=CPU(),
                   compiled_funcs = sample_compiled_funcs,
                   verbose = true,
                   mode="jump",
                   optimizer=MadNLP.Optimizer);


#JuMP, Ipopt
@btime sample_pcfm($ffm, (parameters = $ps, states = $st),
                   $n_samples, 100, heat_constraints!;
                   backend=CPU(),
                   compiled_funcs = sample_compiled_funcs,
                   verbose = true,
                   mode="jump",
                   optimizer=Ipopt.Optimizer);






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

