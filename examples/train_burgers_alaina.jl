# examples/infer_burgers_constrained.jl
using JLD2, PCFM, Lux, Random, Plots

Random.seed!(1234)

println("=" ^ 60)
println("Burgers Equation — Both Constraint Regimes")
println("=" ^ 60)

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

p_loc = 0.5f0
eps_val = 0.02f0
x_grid = Float32.(range(0.0, 1.0, length=nx))
u_0_ic_vec = Float32.(1.0 ./ (1.0 .+ exp.((x_grid .- p_loc) ./ eps_val)))

dx = Float32(1.0 / (nx - 1))
dt_physics = Float32(1.0 / (nt - 1))

# Left BC value
u_L = 0.8f0  # sample from U(0,1)

cdata = make_constraint_data(u_0_ic_vec, nx, nt, n_samples;
    dx=dx, dt_physics=dt_physics, u_L=u_L)

println("  nx=$nx, nt=$nt, dx=$(round(dx;digits=4))")
println("  IC: sigmoid at p_loc=$p_loc")
println("  u_L (Dirichlet) = $u_L")
println("  M₀ = $(round(cdata.M0;digits=4))")

solvers = [
    # Regime 1: BC + Mass
    ("B-BC: BC+Mass (LBFGS)",      BurgersBCMassSolver(penalty=1.0f4)),
    ("B-BC: BC+Mass (IPNewton)",   BurgersBCMassIPSolver()),
    # Regime 2: IC + Mass + Local Flux
    ("B-IC: IC+Flux (LBFGS)",      BurgersICFluxSolver(penalty=1.0f4)),
    # ("B-IC: IC+Flux (IPNewton)",   BurgersICFluxIPSolver()),
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

function mass_conservation_violation(samples, m0, dx)
    _nx, _nt, nc, nb = size(samples)
    viols = [abs(sum(samples[:, k, c, b]) * dx - m0)
             for b in 1:nb, c in 1:nc, k in 1:_nt]
    return (mean=sum(viols)/length(viols), max=maximum(viols))
end

function dirichlet_left_violation(samples, u_L)
    _nx, _nt, nc, nb = size(samples)
    viols = [abs(samples[1, k, c, b] - u_L)
             for b in 1:nb, c in 1:nc, k in 1:_nt]
    return (mean=sum(viols)/length(viols), max=maximum(viols))
end

function neumann_right_violation(samples)
    _nx, _nt, nc, nb = size(samples)
    viols = [abs(samples[end, k, c, b] - samples[end-1, k, c, b])
             for b in 1:nb, c in 1:nc, k in 1:_nt]
    return (mean=sum(viols)/length(viols), max=maximum(viols))
end

function flux_residual_violation(samples, dx, dt_physics)
    _nx, _nt, nc, nb = size(samples)
    viols = Float64[]
    cfl = dt_physics / dx
    for b in 1:nb, c in 1:nc
        for k in 1:(_nt-1)
            u_k = samples[:, k, c, b]
            u_next = samples[:, k+1, c, b]
            # Compute Godunov fluxes from u_k
            for i in 2:(_nx-1)
                F_right = PCFM.godunov_flux(u_k[i], u_k[i+1])
                F_left  = PCFM.godunov_flux(u_k[i-1], u_k[i])
                u_expected = u_k[i] - cfl * (F_right - F_left)
                push!(viols, abs(u_next[i] - u_expected))
            end
        end
    end
    return (mean=sum(viols)/length(viols), max=maximum(viols))
end

println("\n" * "=" ^ 90)
println("BURGERS EQUATION — BOTH REGIMES")
println("=" ^ 90)

println("\n── Regime 1: BC + Mass (H^{B,BC}) ──")
println(rpad("Method", 28), rpad("Dirichlet", 12), rpad("Mass", 12),
        rpad("Neumann", 12), "Time")
println("-" ^ 70)
for r in results
    dv = dirichlet_left_violation(r.samples, u_L)
    mv = mass_conservation_violation(r.samples, cdata.M0, dx)
    nv = neumann_right_violation(r.samples)
    println(rpad(r.label, 28),
            rpad(round(dv.mean;digits=4), 12),
            rpad(round(mv.mean;digits=4), 12),
            rpad(round(nv.mean;digits=4), 12),
            "$(round(r.time;digits=1))s")
end

println("\n── Regime 2: IC + Mass + Local Flux (H^{B,IC}) ──")
println(rpad("Method", 28), rpad("IC", 12), rpad("Mass", 12),
        rpad("Flux resid", 12), "Time")
println("-" ^ 70)
for r in results
    iv = ic_violation(r.samples, u_0_ic_vec)
    mv = mass_conservation_violation(r.samples, cdata.M0, dx)
    fv = flux_residual_violation(r.samples, dx, dt_physics)
    println(rpad(r.label, 28),
            rpad(round(iv.mean;digits=4), 12),
            rpad(round(mv.mean;digits=4), 12),
            rpad(round(fv.mean;digits=4), 12),
            "$(round(r.time;digits=1))s")
end

colors = [:red, :orange, :blue, :purple, :green, :darkgreen]

# Mass over time
p_mass = plot(title="Burgers: Mass m(t) over timesteps", xlabel="k", ylabel="m(t)")
hline!(p_mass, [cdata.M0], label="m₀", lw=2, ls=:dash, color=:black)
for (i, r) in enumerate(results)
    Mk = [sum(r.samples[:, k, 1, 1]) * dx for k in 1:nt]
    plot!(p_mass, 1:nt, Mk, label=r.label, color=colors[i], alpha=0.7)
end
display(p_mass)
savefig("plot_burgers_mass.png")

# Left boundary value over time
p_bc = plot(title="Burgers: u(0,t) left boundary", xlabel="k", ylabel="u(0,t)")
hline!(p_bc, [u_L], label="u_L target", lw=2, ls=:dash, color=:black)
for (i, r) in enumerate(results)
    u_left = [r.samples[1, k, 1, 1] for k in 1:nt]
    plot!(p_bc, 1:nt, u_left, label=r.label, color=colors[i], alpha=0.7)
end
display(p_bc)
savefig("plot_burgers_bc.png")

# IC comparison
p_ic = plot(title="Burgers IC: u(x,0)", xlabel="x", ylabel="u")
plot!(p_ic, x_grid, u_0_ic_vec, label="target", lw=3, ls=:dash, color=:black)
arr = Array(results[3].samples)  # B-IC: IC+Flux
for j in 1:min(2, n_samples)
    plot!(p_ic, x_grid, arr[:,1,1,j], label="IC+Flux s$j", alpha=0.5)
end
display(p_ic)
savefig("plot_burgers_ic.png")

println("\nDone.")