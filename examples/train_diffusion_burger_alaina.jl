using JLD2, PCFM, Lux, Random, Plots

# Load trained model
saved = JLD2.load("examples/checkpoints/ffm_burgers_checkpoint.jld2")
ps = saved["parameters"]
st = saved["states"]

n_samples = 4
n_steps = 20

nx, nt = 100, 100
emb_channels = 32

ffm = FFM(nx=nx, nt=nt, emb_channels=emb_channels,
          hidden_channels=64, proj_channels=256,
          n_layers=4, modes=(32,32), device=cpu_device())
_, st_inf = Lux.setup(Random.default_rng(), ffm.model)

# Burgers constraint data
dx = Float32(1.0 / (nx - 1))       # grid spacing on [0,1] with nx points
dt_physics = Float32(1.0 / (nt - 1))

# IC 
p_loc = 0.5f0
eps   = 0.02f0
x_grid = Float32.(range(0.0, 1.0, length=nx))
u_0_ic_vec = Float32.(1.0 ./ (1.0 .+ exp.((x_grid .- p_loc) ./ eps)))

cdata = make_constraint_data(u_0_ic_vec, nx, nt, n_samples;
    dx=dx, dt_physics=dt_physics)


solvers = [
    # ("Unconstrained",           nothing),
    # ("IC-only (analytic)",      AnalyticICProjectionSolver()),
    ("B4: Burgers (LBFGS)",     BurgersConservationSolver(penalty=1.0f4)),
    ("B4: Burgers (IPNewton)",  BurgersIPNewtonSolver()),
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

function ic_violation(samples, u_ic)
    _nx, _nt, nc, nb = size(samples)
    viols = [sqrt(sum(abs2, samples[:, 1, c, b] .- u_ic)) for b in 1:nb, c in 1:nc]
    return (mean=sum(viols)/length(viols), max=maximum(viols))
end

function burgers_flux_violation(samples)
    _nx, _nt, nc, nb = size(samples)
    viols = [abs(samples[end, k, c, b]^2 - samples[1, k, c, b]^2)
             for b in 1:nb, c in 1:nc, k in 1:_nt]
    return (mean=sum(viols)/length(viols), max=maximum(viols))
end

function burgers_mass_violation(samples, dx, dt_physics)
    _nx, _nt, nc, nb = size(samples)
    viols = Float64[]
    for b in 1:nb, c in 1:nc
        for k in 1:(_nt-1)
            u_k = samples[:, k, c, b]
            M_k = sum(u_k) * dx
            flux = -0.5f0 * (u_k[end]^2 - u_k[1]^2)
            M_target = M_k + dt_physics * flux
            M_next = sum(samples[:, k+1, c, b]) * dx
            push!(viols, abs(M_next - M_target))
        end
    end
    return (mean=sum(viols)/length(viols), max=maximum(viols))
end

println("\n" * "=" ^ 70)
println("BURGERS EQUATION METRICS")
println("=" ^ 70)
println()
println(rpad("Method", 28), rpad("IC viol", 12), rpad("Flux viol", 12),
        rpad("Mass viol", 12), "Time")
println("-" ^ 70)

for r in results
    iv = ic_violation(r.samples, u_0_ic_vec)
    fv = burgers_flux_violation(r.samples)
    mv = burgers_mass_violation(r.samples, dx, dt_physics)
    println(rpad(r.label, 28), rpad(round(iv.mean;digits=4), 12),
            rpad(round(fv.mean;digits=4), 12),
            rpad(round(mv.mean;digits=4), 12),
            "$(round(r.time;digits=1))s")
end

p = plot(title="Burgers IC: u(x,0)", xlabel="x", ylabel="u")
plot!(p, x_grid, u_0_ic_vec, label="target IC", lw=3, ls=:dash, color=:black)
if length(results) >= 3
    arr = Array(results[3].samples)
    for i in 1:n_samples
        plot!(p, x_grid, arr[:,1,1,i], label="constrained $i", alpha=0.6)
    end
end
display(p)

println("\nDone.")