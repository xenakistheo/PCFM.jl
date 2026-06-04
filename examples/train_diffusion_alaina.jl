using JLD2, FFTW, Plots, Functors, Random
using PCFM, Lux

Random.seed!(1234)

batch_size = 32
nx, nt = 100, 100
emb_channels = 32
n_epochs = 1000
n_samples = 4
n_steps = 20

# Data generation parameters
visc_range = (1.0f0, 5.0f0)
phi_range = (0.0f0, Float32(π))
t_range = (0.0f0, 1.0f0)

println("=" ^ 60)
println("PCFM Full Evaluation — All Projection Solvers")
println("=" ^ 60)

println("\n[1] Generating data …")
u_data = generate_diffusion_data(batch_size, nx, nt,
    (1.0f0, 5.0f0), (0.0f0, Float32(π)), (0.0f0, 1.0f0))

println("\n[2] Creating model …")
ffm = FFM(nx=nx, nt=nt, emb_channels=emb_channels,
          hidden_channels=64, proj_channels=256,
          n_layers=4, modes=(32,32), device=cpu_device())

weight_file = "heat_eq_weights.jld2"
if isfile(weight_file)
    println("\n[3] Loading saved weights …")
    saved = JLD2.load(weight_file)
    ps, st = saved["parameters"], saved["states"]
else
    println("\n[3] Training …")
    compiled_funcs = PCFM.compile_functions(ffm, batch_size)
    losses, tstate = train_ffm!(ffm, u_data;
        compiled_funcs=compiled_funcs, epochs=n_epochs, verbose=true)
    ps = fmap(x -> x isa AbstractArray ? Array(x) : x, tstate.parameters)
    st = fmap(x -> x isa AbstractArray ? Array(x) : x, tstate.states)
    JLD2.save(weight_file, "parameters", ps, "states", st, "config", ffm.config)
end
_, st_inf = Lux.setup(Random.default_rng(), ffm.model)

# Constraint data

x_grid = range(0, 2π, length=nx)
u_0_ic_vec = Float32.(sin.(x_grid .+ π/4))
dx = Float32(2π / nx)
cdata = make_constraint_data(u_0_ic_vec, nx, nt, n_samples; dx=dx)
E0 = sum(abs2, u_0_ic_vec) * dx
M0 = cdata.M0

println("\nPhysics targets:")
println("  E₀ (energy) = $(round(E0; digits=4))")
println("  M₀ (mass)   = $(round(M0; digits=4))")

# solvers to test - need to add to this

solvers = [
    ("Unconstrained",            NoOpSolver()),
    ("IC-only (analytic)",       AnalyticICProjectionSolver()),
    ("Energy-only (analytic)",   AnalyticEnergyProjectionSolver()),
    ("IC+Energy (analytic)",     AnalyticICEnergyProjectionSolver()),
    ("Mass-only (analytic)",     AnalyticMassProjectionSolver()),
    ("IC+Mass (analytic)",       AnalyticICMassProjectionSolver()),
    ("IC+Energy (LBFGS)",        PenaltyLBFGSEnergyProjectionSolver(penalty=1.0f4)),
    ("IC+Mass (LBFGS)",          PenaltyLBFGSMassProjectionSolver(penalty=1.0f4)),
    ("Energy (Interior Point)",             IPEnergyProjectionSolver()),
    ("Mass (Interior Point)",             IPMassProjectionSolver()),

    # ── Eq 14: Reaction-Diffusion ──
    # ("Eq14: Reaction-Diffusion (LBFGS)",         RDSolver(penalty=1.0f4, rho=1.0f0)),
    # ("Eq14: Reaction-Diffusion (IPNewton)",      RDIPNewtonSolver(rho=1.0f0)),

    # # ── B2: NS Vorticity ──
    # ("B2: Vorticity (analytic)",         NSVorticityAnalyticSolver()),
    # ("B2: Vorticity (LBFGS)",           NSVorticityLBFGSSolver(penalty=1.0f4)),
    # ("B2: Vorticity (IPNewton)",        NSVorticityIPNewtonSolver()),

    # # ── B4: Burgers ──
    # ("B4: Burgers (LBFGS)",             BurgersConservationSolver(penalty=1.0f4)),
    # ("B4: Burgers (IPNewton)",          BurgersIPNewtonSolver()),
]

# Run all solvers and collect results

results = []

for (i, (label, solver)) in enumerate(solvers)
    println("\n[$(i+3)] $label …")
    t0 = time()

    if solver === NoOpSolver()
        # Unconstrained
        samples = sample_pcfm(ffm.model, ps, st_inf, nx, nt, emb_channels, n_samples, n_steps, solver, cdata, verbose=false)
        # samples = sample_ffm(ffm, (ps, st_inf), n_samples, n_steps; verbose=false)
    else
        samples = sample_pcfm(ffm.model, ps, st_inf, nx, nt, emb_channels,
            n_samples, n_steps, solver, cdata; verbose=false)
    end

    dt = time() - t0
    println("  $(round(dt; digits=1)) s")
    push!(results, (label=label, samples=samples, time=dt))
end

# Metric functions

function ic_violation(samples, u_ic)
    _nx, _nt, nc, nb = size(samples)
    viols = Float64[]
    for b in 1:nb, c in 1:nc
        push!(viols, sqrt(sum(abs2, samples[:, 1, c, b] .- u_ic)))
    end
    return (mean = sum(viols)/length(viols), max = maximum(viols))
end

function energy_violation(samples, E0, dx)
    _nx, _nt, nc, nb = size(samples)
    viols = Float64[]
    for b in 1:nb, c in 1:nc, k in 1:_nt
        Ek = sum(abs2, samples[:, k, c, b]) * dx
        push!(viols, abs(Ek - E0))
    end
    return (mean = sum(viols)/length(viols), max = maximum(viols))
end

function mass_violation(samples, dx)
    _nx, _nt, nc, nb = size(samples)
    viols = Float64[]
    for b in 1:nb, c in 1:nc, k in 1:_nt
        Mk = sum(samples[:, k, c, b]) * dx
        push!(viols, abs(Mk))   # target is 0 for heat eq
    end
    return (mean = sum(viols)/length(viols), max = maximum(viols))
end

# Print all metrics

println("\n" * "=" ^ 80)
println("METRICS")
println("=" ^ 80)

println("\n", rpad("Method", 28),
        rpad("IC viol", 14),
        rpad("Energy viol", 14),
        rpad("Mass viol", 14),
        "Time")
println("-" ^ 80)

for r in results
    iv = ic_violation(r.samples, u_0_ic_vec)
    ev = energy_violation(r.samples, E0, dx)
    mv = mass_violation(r.samples, dx)

    println(rpad(r.label, 28),
            rpad(string(round(iv.mean; digits=4)), 14),
            rpad(string(round(ev.mean; digits=4)), 14),
            rpad(string(round(mv.mean; digits=4)), 14),
            "$(round(r.time; digits=1))s")
end

# ═══════════════════════════════════════════════════════════
#  Plots
# ═══════════════════════════════════════════════════════════

# Plot 1: IC slice for IC+Energy
# arr_ie = Array(results[4].samples)  # IC+Energy analytic
# p_ic = plot(title="IC: u(x,0) — IC+Energy projection", xlabel="x", ylabel="u")
# plot!(p_ic, x_grid, u_0_ic_vec, label="target IC", lw=3, ls=:dash, color=:black)
# for i in 1:n_samples
#     plot!(p_ic, x_grid, arr_ie[:,1,1,i], label="sample $i", alpha=0.6)
# end
# display(p_ic)
# savefig("plot_ic.png")

# Plot 2: Energy over time for all methods
p_energy = plot(title="Energy E(t) = ∫u²dx across timesteps",
                xlabel="timestep k", ylabel="E(t)", legend=:outerright)
hline!(p_energy, [E0], label="E₀", lw=2, ls=:dash, color=:black)

colors = [:red, :orange, :blue, :green, :purple, :cyan, :darkgreen, :darkcyan, :magenta, :brown, :black, :steelblue, :lightyellow, :azure, :pink, :sienna, :grey0]
for (i, r) in enumerate(results)
    Ek = [sum(abs2, r.samples[:, k, 1, 1]) * dx for k in 1:nt]
    plot!(p_energy, 1:nt, Ek, label=r.label, color=colors[i], alpha=0.7)
end
display(p_energy)
savefig("plot_energy.png")

# Plot 3: Mass over time for all methods
p_mass = plot(title="Mass M(t) = ∫u dx across timesteps",
              xlabel="timestep k", ylabel="M(t)", legend=:outerright)
hline!(p_mass, [0.0], label="M₀=0", lw=2, ls=:dash, color=:black)

for (i, r) in enumerate(results)
    Mk = [sum(r.samples[:, k, 1, 1]) * dx for k in 1:nt]
    plot!(p_mass, 1:nt, Mk, label=r.label, color=colors[i], alpha=0.7)
end
display(p_mass)
savefig("plot_mass.png")

# Plot 4: Heatmaps comparing unconstrained vs IC+Energy
p_heatmaps = plot(
    heatmap(Array(results[1].samples)[:,:,1,1], title="Unconstrained", c=:viridis),
    heatmap(Array(results[4].samples)[:,:,1,1], title="IC+Energy (analytic)", c=:viridis),
    heatmap(Array(results[6].samples)[:,:,1,1], title="IC+Mass (analytic)", c=:viridis),
    layout=(1,3), size=(1200, 300)
)
display(p_heatmaps)
savefig("plot_heatmaps.png")

println("\nDone.")