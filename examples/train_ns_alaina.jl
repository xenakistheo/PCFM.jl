# examples/train_ns.jl
using JLD2, PCFM, Lux, Random, Plots

Random.seed!(1234)

println("=" ^ 60)
println("B2: NS Vorticity — Constrained Sampling (reduced grid)")
println("=" ^ 60)

s = 32       
nt_ns = 50      # fewer timesteps
emb_channels = 32
n_samples = 4 
n_steps = 20

println("  Grid: $(s)×$(s)×$(nt_ns) = $(s*s) vars/slice")
println("  Total sub-problems per run: $(nt_ns) × $(n_samples) × $(n_steps) = $(nt_ns * n_samples * n_steps)")

ffm = FFM(spatial_size=(s, s), nt=nt_ns, emb_channels=emb_channels,
          hidden_channels=32, proj_channels=128,
          n_layers=2, modes=(4, 4, 4), device=cpu_device())
ps, st_inf = Lux.setup(Random.default_rng(), ffm.model)

x_grid = Float32.(range(0.0, 1.0, length=s))
u_0_ic = Float32.([sin(2π*x + 2π*y) for x in x_grid, y in x_grid])
dx = Float32(1.0 / s)

cdata = make_constraint_data(u_0_ic, (s, s), nt_ns, n_samples; dx=dx)
W0 = cdata.M0

println("  W₀ = $(round(W0; digits=4))")
println("  dx = $(round(dx; digits=4))")

solvers = [
    ("B2: Vorticity (analytic)",   NSVorticityAnalyticSolver()),
    ("B2: Vorticity (LBFGS)",     NSVorticityLBFGSSolver(penalty=1.0f4)),
    ("B2: Navier Stokes (IP)", NSVorticityIPNewtonSolver())
]

results = []
for (i, (label, solver)) in enumerate(solvers)
    println("\n[$i] $label …")
    t0 = time()
    samples = sample_pcfm(ffm.model, ps, st_inf, (s, s), nt_ns, emb_channels,
        n_samples, n_steps, solver, cdata; verbose=false)
    dt = time() - t0
    println("  $(round(dt; digits=1)) s")
    push!(results, (label=label, samples=samples, time=dt))
end

function vorticity_violation(samples, W0, dx)
    layout = PCFM.get_array_layout(samples)
    viols = Float64[]
    for b in 1:layout.nb, c in 1:layout.nc, k in 1:layout.nt
        slice = PCFM.get_slice(samples, k, c, b)
        push!(viols, abs(sum(slice) * dx - W0))
    end
    return (mean=sum(viols)/length(viols), max=maximum(viols))
end

println("\n" * "=" ^ 70)
println("B2: NS VORTICITY METRICS ($(s)×$(s), reduced)")
println("=" ^ 70)
println()
println(rpad("Method", 30), rpad("Vort. viol (mean)", 20), "Time")
println("-" ^ 60)

for r in results
    vv = vorticity_violation(r.samples, W0, dx)
    println(rpad(r.label, 30), rpad(round(vv.mean; digits=6), 20),
            "$(round(r.time;digits=1))s")
end

println("\nDone.")