"""
Example script for training Functional Flow Matching on 1D diffusion equation.

This script demonstrates:
1. Creating an FFM (Functional Flow Matching) model
2. Generating training data
3. Compiling functions with Reactant
4. Training the model
5. Generating samples
"""

using PCFM
using Reactant, Lux
using JLD2, Functors
using Plots

# Set random seed
using Random
Random.seed!(1234)

# Configuration
batch_size = 32

nx = 100          # Spatial resolution
nt = 100          # Temporal resolution
emb_channels = 32
n_epochs = 1000
force_retrain = false

# Checkpoint path
weight_file = joinpath(@__DIR__, "checkpoints", "ffm_diffusion_checkpoint.jld2")

# Data generation parameters
visc_range = (1.0f0, 5.0f0)
phi_range = (0.0f0, Float32(π))
t_range = (0.0f0, 1.0f0)

println("=" ^ 60)
println("Training Functional Flow Matching on 1D Diffusion Equation")
println("=" ^ 60)

# 1. Generate training data
println("\n[1/5] Generating training data...")
u_data = generate_diffusion_data(batch_size, nx, nt, visc_range, phi_range, t_range)
println("  Data shape: $(size(u_data))")

# 2. Create model
println("\n[2/5] Creating FFM (Functional Flow Matching) model...")
ffm = FFM(
    nx = nx,
    nt = nt,
    emb_channels = emb_channels,
    hidden_channels = 64,
    proj_channels = 256,
    n_layers = 4,
    modes = (32, 32),
    device = reactant_device()
);
println("  Model created successfully")

# 3. Load checkpoint (if available) or train model
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

# 5. Generate samples
println("\n[5/5] Generating samples...")
n_samples = 32
# Reuse compiled_funcs if n_samples == batch_size, otherwise compile for n_samples
sample_compiled_funcs = (n_samples == batch_size) ? compiled_funcs : PCFM.compile_functions(ffm, n_samples)

# samples = sample_pcfm(ffm, (parameters = ps, states = st), n_samples, 100;
    # compiled_funcs = sample_compiled_funcs, verbose = true)

# p = (nx=Nx, nt=nt, dx=dx, u0=vec(u1[:,1]), backend=backend)
samples_exa = sample_pcfm(ffm, (parameters = ps, states = st), n_samples, 100, heat_constraints!, nothing; compiled_funcs = sample_compiled_funcs, verbose = true)


println("\n" * "=" ^ 60)
println("Training Complete!")
println("=" ^ 60)

# Visualize results
println("\nPlotting results...")

# Plot training curve
if !isempty(losses)
    p1 = plot(1:length(losses), losses,
        yscale = :log10,
        xlabel = "Epoch",
        ylabel = "Loss (log scale)",
        title = "Training Loss",
        legend = false,
        linewidth = 2)
    display(p1)
end

# Plot samples
arr_data = Array(u_data)
arr_samples = Array(samples)

p_data = [Plots.heatmap(arr_data[:, :, 1, i],
              title = "Training Data $i",
              xlabel = "Time",
              ylabel = "Space",
              c = :viridis)
          for i in 1:min(2, batch_size)]

p_samples = [Plots.heatmap(arr_samples[:, :, 1, i],
                 title = "Generated Sample $i",
                 xlabel = "Time",
                 ylabel = "Space",
                 c = :viridis)
             for i in 1:2]

# Combine plots
p2 = Plots.plot(p_data..., layout = (1, length(p_data)), size = (800, 300))
p3 = Plots.plot(p_samples..., layout = (1, length(p_samples)), size = (800, 300))

display(p2)
display(p3)

println("\nDone! Check the plots above.")
