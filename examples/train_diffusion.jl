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
batch_size    = 32
nx            = 100     # Spatial resolution
nt            = 100     # Temporal resolution
emb_channels  = 32
n_epochs      = 1000
force_retrain = false

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

# 3. Compile functions (optional but recommended for speed)
println("\n[3/5] Compiling functions with Reactant...")
compiled_funcs = PCFM.compile_functions(ffm, batch_size)

# 4. Train or load checkpoint
losses = Float32[]
if isfile(weight_file) && !force_retrain
    println("\n[4/5] Loading checkpoint from: $weight_file")
    saved  = JLD2.load(weight_file)
    device = ffm.config[:device]
    ps     = saved["parameters"] |> device
    st     = saved["states"]     |> device
    losses = saved["losses"]
    println("  Loaded parameters, states, and loss history")
else
    println("\n[4/5] Training model for $n_epochs epochs...")
    losses, tstate = train_ffm!(ffm, u_data; compiled_funcs, epochs = n_epochs, verbose = true)
    println("\nFinal loss: $(losses[end])")

    ps = fmap(x -> x isa AbstractArray ? Array(x) : x, tstate.parameters)
    st = fmap(x -> x isa AbstractArray ? Array(x) : x, tstate.states)
    mkpath(dirname(weight_file))
    JLD2.save(weight_file, "parameters", ps, "states", st, "losses", losses, "config", ffm.config)
    println("Checkpoint saved to: $weight_file")
end

# 5. Plot training curve
p1 = plot(1:length(losses), losses,
    yscale = :log10,
    xlabel = "Epoch", ylabel = "Loss (log scale)",
    title = "Diffusion — Training Loss",
    legend = false, linewidth = 2)
display(p1)

println("\nDone!")
