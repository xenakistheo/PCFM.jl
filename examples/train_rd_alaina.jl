# examples/infer_rd_constrained.jl
using JLD2, PCFM, Lux, Random, Plots

Random.seed!(1234)

println("=" ^ 60)
println("Reaction-Diffusion — Constrained Sampling")
println("=" ^ 60)

saved = JLD2.load("examples/checkpoints/ffm_rd_checkpoint.jld2")
ps = saved["parameters"]
st = saved["states"]

# RD typically uses Neumann BCs on [0,1]
nx, nt = 100, 100
emb_channels = 32
n_samples = 4
n_steps = 20

ffm = FFM(nx=nx, nt=nt, emb_channels=emb_channels,
          hidden_channels=64, proj_channels=256,
          n_layers=4, modes=(32,32), device=cpu_device())
_, st_inf = Lux.setup(Random.default_rng(), ffm.model)

# IC
x_grid = Float32.(range(0.0, 1.0, length=nx))
u_0_ic_vec = Float32.(exp.(-((x_grid .- 0.5f0) ./ 0.1f0).^2)) #TODO

dx = Float32(1.0 / (nx - 1))
dt_phys = Float32(1.0 / (nt - 1))

cdata = make_constraint_data(u_0_ic_vec, nx, nt, n_samples;
    dx=dx, dt_physics=dt_phys)

rho = 1.0f0  # reaction rate — CHECK WHAT VALUE TO USE

println("  nx=$nx, nt=$nt")
println("  IC: Gaussian bump at x=0.5")
println("  ρ (reaction rate) = $rho")
println("  M₀ = $(round(cdata.M0;digits=4))")
println("  BCs: Neumann (zero flux), g_L=g_R=0")

# ── Solvers ──
solvers = [
    # ("Unconstrained",                nothing),
    # ("IC-only (analytic)",           AnalyticICProjectionSolver()),
    ("Eq14: Reaction-Diffusion (LBFGS)",    RDSolver(penalty=1.0f4, rho=rho)),
    ("Eq14: Reaction-Diffusion (IPNewton)",  RDIPNewtonSolver(rho=rho)),
]

results = []
for (i, (label, solver)) in enumerate(solvers)
    println("\n[$i] $label …")
    t0 = time()
    samples = sample_pcfm(ffm.model, ps, st_inf, nx, nt, emb_channels,
            n_samples, n_steps, solver, cdata; verbose=false)
    dt = time() - t0
    println("  $(round(dt; digits=1)) s")
    push!(results, (label=label, samples=samples, time=dt))
end

# ── Metrics ──
function ic_violation(samples, u_ic)
    _nx, _nt, nc, nb = size(samples)
    viols = [sqrt(sum(abs2, samples[:, 1, c, b] .- u_ic)) for b in 1:nb, c in 1:nc]
    return (mean=sum(viols)/length(viols), max=maximum(viols))
end

function rd_mass_violation(samples, dx, dt_phys, rho)
    _nx, _nt, nc, nb = size(samples)
    viols = Float64[]
    for b in 1:nb, c in 1:nc
        for k in 1:(_nt-1)
            u_k = samples[:, k, c, b]
            M_k = sum(u_k) * dx
            # Source: ρ ∫u(1-u)dx
            source = rho * sum(u_k[i] * (1.0f0 - u_k[i]) for i in 1:_nx) * dx
            # No boundary flux (Neumann BCs)
            M_target = M_k + dt_phys * source
            M_next = sum(samples[:, k+1, c, b]) * dx
            push!(viols, abs(M_next - M_target))
        end
    end
    return (mean=sum(viols)/length(viols), max=maximum(viols))
end

println("\n" * "=" ^ 70)
println("RD METRICS")
println("=" ^ 70)
println()
println(rpad("Method", 32), rpad("IC viol", 12),
        rpad("RD mass viol", 18), "Time")
println("-" ^ 70)

for r in results
    iv = ic_violation(r.samples, u_0_ic_vec)
    fv = rd_mass_violation(r.samples, dx, dt_phys, rho)
    println(rpad(r.label, 32), rpad(round(iv.mean;digits=4), 12),
            rpad(round(fv.mean;digits=6), 18),
            "$(round(r.time;digits=1))s")
end

# ── Plot ──
p = plot(title="RD IC: u(x,0)", xlabel="x", ylabel="u")
plot!(p, x_grid, u_0_ic_vec, label="target IC", lw=3, ls=:dash, color=:black)
if length(results) >= 3
    arr = Array(results[3].samples)
    for i in 1:n_samples
        plot!(p, x_grid, arr[:,1,1,i], label="constrained $i", alpha=0.6)
    end
end
display(p)

println("\nDone.")