# """
# Example script for training Functional Flow Matching on 1D diffusion equation.

# This script demonstrates:
# 1. Creating an FFM (Functional Flow Matching) model
# 2. Generating training data
# 3. Compiling functions with Reactant
# 4. Training the model
# 5. Generating samples
# """

# using JLD2
# using FFTW
# using PCFM
# using Lux #Reactant, Lux
# using Plots
# using Functors

# # Set random seed
# using Random
# Random.seed!(1234)

# # Configuration
# batch_size = 32
# nx = 100          # Spatial resolution
# nt = 100          # Temporal resolution
# emb_channels = 32
# n_epochs = 1000

# n_samples    = 8          # keep small for quick eval
# n_steps      = 100

# # Data generation parameters
# visc_range = (1.0f0, 5.0f0)
# phi_range = (0.0f0, Float32(π))
# t_range = (0.0f0, 1.0f0)

# println("=" ^ 60)
# println("Training Functional Flow Matching on 1D Diffusion Equation")
# println("=" ^ 60)

# # 1. Generate training data
# println("\n[1/5] Generating training data...")
# u_data = generate_diffusion_data(batch_size, nx, nt, visc_range, phi_range, t_range)
# println("  Data shape: $(size(u_data))")

# # 2. Create model
# println("\n[2/5] Creating FFM (Functional Flow Matching) model...")
# ffm = FFM(
#     nx = nx,
#     nt = nt,
#     emb_channels = emb_channels,
#     hidden_channels = 64,
#     proj_channels = 256,
#     n_layers = 4,
#     modes = (32, 32),
#     device = cpu_device()
# );
# println("  Model created successfully")

# # # 3. Compile functions (optional but recommended for speed)
# # println("\n[3/5] Compiling functions with Reactant...")
# # compiled_funcs = PCFM.compile_functions(ffm, batch_size)

# # println("\n[3/5] Compiling functions with Reactant...")
# weight_file = "heat_eq_weights.jld2"
# if isfile(weight_file)
#     println("\nFound saved weights, skipping training...")
#     saved = JLD2.load(weight_file)
#     tstate = (parameters = saved["parameters"], states = saved["states"])
#     losses = Float32[]   # empty placeholder since no training
# else
#     # 4. Train model
#     println("\n[4/5] Training model for $n_epochs epochs...")
#     losses, tstate = train_ffm!(ffm, u_data; compiled_funcs, epochs = n_epochs, verbose = true)
#     # Convert Reactant arrays → plain Julia arrays before saving
#     ps_plain = fmap(x -> x isa AbstractArray ? Array(x) : x, tstate.parameters)
#     st_plain = fmap(x -> x isa AbstractArray ? Array(x) : x, tstate.states)
#     JLD2.save(weight_file, "parameters", ps_plain, #tstate.parameters,
#                            "states",     st_plain, #tstate.states,
#                            "config",     ffm.config)
# end

# if !isempty(losses)
#     println("\nFinal loss: $(losses[end])")
# end

# # 5. Generate samples
# println("\n[5/5] Generating samples...")
# n_samples = 32
# # samples = sample_ffm(ffm, tstate, n_samples, 100; compiled_funcs, verbose = true)

# # ps_cpu = fmap(Array, tstate.parameters)
# # st_cpu = fmap(Array, tstate.states)
# # tstate_cpu = (parameters = ps_cpu, states = st_cpu)
# # samples = sample_pcfm(ffm, tstate, n_samples, 100; verbose = true)
# # Extract what we need from ffm directly
# nx = ffm.config[:nx]
# nt = ffm.config[:nt]
# emb_channels = ffm.config[:emb_channels]

# # Use tstate loaded from JLD2 (already plain Arrays)
# ps = tstate.parameters
# st = tstate.states
# # Reinitialize model states for CPU inference
# _, st_cpu = Lux.setup(Random.default_rng(), ffm.model)

# samples = sample_pcfm_cpu(ffm.model, ps, st, nx, nt, emb_channels, n_samples, 100; verbose=true)



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

# p_data = [heatmap(arr_data[:, :, 1, i],
#               title = "Training Data $i",
#               xlabel = "Time",
#               ylabel = "Space",
#               c = :viridis)
#           for i in 1:min(2, batch_size)]

# p_samples = [heatmap(arr_samples[:, :, 1, i],
#                  title = "Generated Sample $i",
#                  xlabel = "Time",
#                  ylabel = "Space",
#                  c = :viridis)
#              for i in 1:2]

# # Combine plots
# p2 = plot(p_data..., layout = (1, length(p_data)), size = (800, 300))
# p3 = plot(p_samples..., layout = (1, length(p_samples)), size = (800, 300))

# #display(p1)
# display(p2)
# display(p3)

# println("\nSaving model weights...")
# JLD2.save("heat_eq_weights.jld2",
#     "parameters", tstate.parameters,
#     "states",     tstate.states,
#     "config",     ffm.config)
# println("Weights saved to heat_eq_weights.jld2")

# println("\nDone! Check the plots above.")
"""
Evaluation script: nonlinear (energy-conservation) constraint benchmark.

Demonstrates that the linear IC projection is *insufficient* for
energy conservation, while the Optimization.jl IPNewton backend
handles the quadratic constraint correctly.
"""

using JLD2, FFTW, Plots, Functors, Random
using PCFM
using Lux

Random.seed!(1234)

# ──────────────── configuration ────────────────
batch_size   = 32
nx, nt       = 100, 100
emb_channels = 32
n_epochs     = 1000
n_samples    = 8          # keep small for quick eval
n_steps      = 100

visc_range = (1.0f0, 5.0f0)
phi_range  = (0.0f0, Float32(π))
t_range    = (0.0f0, 1.0f0)

println("=" ^ 60)
println("PCFM Evaluation — Nonlinear Energy-Conservation Constraint")
println("=" ^ 60)

# ───────── 1. data ─────────
println("\n[1/6] Generating training data …")
u_data = generate_diffusion_data(batch_size, nx, nt, visc_range, phi_range, t_range)
println("  Data shape: $(size(u_data))")

# 2 - model
println("\n[2/6] Creating FFM model …")
ffm = FFM(
    nx = nx, nt = nt,
    emb_channels = emb_channels,
    hidden_channels = 64,
    proj_channels = 256,
    n_layers = 4,
    modes = (32, 32),
    device = cpu_device()
)

# 3 - train or load 
weight_file = "heat_eq_weights.jld2"
if isfile(weight_file)
    println("\n[3/6] Loading saved weights …")
    saved = JLD2.load(weight_file)
    ps = saved["parameters"]
    st = saved["states"]
else
    println("\n[3/6] Training for $n_epochs epochs …")
    compiled_funcs = PCFM.compile_functions(ffm, batch_size)
    losses, tstate = train_ffm!(ffm, u_data;
                                compiled_funcs = compiled_funcs,
                                epochs = n_epochs, verbose = true)
    ps = fmap(x -> x isa AbstractArray ? Array(x) : x, tstate.parameters)
    st = fmap(x -> x isa AbstractArray ? Array(x) : x, tstate.states)
    JLD2.save(weight_file,
              "parameters", ps, "states", st, "config", ffm.config)
    println("  Final loss: $(losses[end])")
end

# Re-init Lux states for inference
_, st_inf = Lux.setup(Random.default_rng(), ffm.model)

# 4- define IC + constraints
x_grid     = range(0, 2π, length = nx)
u_0_ic_vec = Float32.(sin.(x_grid .+ π/4))
dx         = Float32(2π / nx)

# (a) linear IC-only constraint  (baseline — provably insufficient for energy)
ic_cons!, n_ic = ic_constraint(u_0_ic_vec, nx, nt)

# (b) nonlinear: IC + energy conservation
ie_cons!, n_ie = ic_and_energy_constraint(u_0_ic_vec, nx, nt; dx = dx)

# (c) energy-only (to isolate the nonlinear part)
en_cons!, n_en = energy_constraint(u_0_ic_vec, nx, nt; dx = dx)

# ───────── 5. sample with each backend / constraint ─────────
solver_ip  = OptimizationIPNewtonSolver()
solver_lbfgs = OptimizationLBFGSSolver(; penalty = 1.0f6)

println("\n[4/6] Sampling — unconstrained FFM …")
t0 = time()
samples_ffm = sample_ffm(ffm, (ps, st_inf), n_samples, n_steps; verbose = false)
dt_ffm = time() - t0
println("  wall time: $(round(dt_ffm; digits=2)) s")

println("\n[5/6] Sampling — PCFM with IC-only (linear) via IPNewton …")
t0 = time()
samples_ic = sample_pcfm_final(
    ffm.model, ps, st_inf, nx, nt, emb_channels,
    n_samples, n_steps,
    solver_ip, ic_cons!, n_ic;
    verbose = false
)
dt_ic = time() - t0
println("  wall time: $(round(dt_ic; digits=2)) s")

println("\n[6/6] Sampling — PCFM with IC+Energy (nonlinear) via IPNewton …")
t0 = time()
samples_ie = sample_pcfm_final(
    ffm.model, ps, st_inf, nx, nt, emb_channels,
    n_samples, n_steps,
    solver_ip, ie_cons!, n_ie;
    verbose = false
)
dt_ie = time() - t0
println("  wall time: $(round(dt_ie; digits=2)) s")

# ───────── 6. evaluate metrics ─────────
println("\n" * "=" ^ 60)
println("METRICS")
println("=" ^ 60)

# Helper: compute constraint violation
function constraint_violation(samples, cons!, n_cons)
    _nx, _nt, nc, nb = size(samples)
    max_viol = 0.0
    mean_viol = 0.0
    count = 0
    res = zeros(Float32, n_cons)
    for b in 1:nb, c in 1:nc
        u = vec(samples[:, :, c, b])
        cons!(res, u, (u,))      # p[1] unused for most constraints
        v = sqrt(sum(abs2, res))
        max_viol = max(max_viol, v)
        mean_viol += v
        count += 1
    end
    return mean_viol / count, max_viol
end

# Energy constraint violation across all three sample sets
mean_v_ffm, max_v_ffm = constraint_violation(samples_ffm, en_cons!, n_en)
mean_v_ic,  max_v_ic  = constraint_violation(samples_ic,  en_cons!, n_en)
mean_v_ie,  max_v_ie  = constraint_violation(samples_ie,  en_cons!, n_en)

println("\nEnergy-conservation violation  ‖h(z)‖ :")
println("  Unconstrained FFM    — mean: $(round(mean_v_ffm;digits=4)),  max: $(round(max_v_ffm;digits=4))")
println("  PCFM (IC-only)       — mean: $(round(mean_v_ic; digits=4)),  max: $(round(max_v_ic; digits=4))")
println("  PCFM (IC + Energy)   — mean: $(round(mean_v_ie; digits=4)),  max: $(round(max_v_ie; digits=4))")

println("\nWall time (projection-dominated):")
println("  FFM (no projection) : $(round(dt_ffm; digits=2)) s")
println("  PCFM IC-only        : $(round(dt_ic;  digits=2)) s")
println("  PCFM IC+Energy      : $(round(dt_ie;  digits=2)) s")

# plots
arr = Array(samples_ie)
p_samples = [heatmap(arr[:, :, 1, i],
                 title = "IC+Energy Sample $i",
                 xlabel = "Time", ylabel = "Space", c = :viridis)
             for i in 1:min(4, n_samples)]
display(plot(p_samples..., layout = (1, length(p_samples)),
             size = (300 * length(p_samples), 300)))

# Check IC satisfaction visually
p_ic = plot(title = "IC slice u(x, 0)", xlabel = "x", ylabel = "u")
plot!(p_ic, x_grid, u_0_ic_vec, label = "target IC", lw = 3, ls = :dash)
for i in 1:min(4, n_samples)
    plot!(p_ic, x_grid, arr[:, 1, 1, i], label = "sample $i", alpha = 0.6)
end
display(p_ic)

println("\nDone.")