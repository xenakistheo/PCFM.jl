"""
Test constraint violation for different 
floating point precisions, using 
JuMP + Ipopt as the solver.
"""


using JuMP, Ipopt
using KernelAbstractions
using LinearAlgebra

backend = CPU()

nx        = 10
nt        = 5
n_samples = 1
N         = nx * nt * n_samples

x_grid  = range(0.0, 2.0*π; length = nx)           # Float64
u0_ic   = sin.(x_grid .+ π/4)                       # Float64
u0_mat  = KernelAbstractions.adapt(backend, repeat(reshape(u0_ic, nx, 1), 1, n_samples))  # CuArray{Float64}

idx(i, t, s) = i + (t-1)*nx + (s-1)*nx*nt

function run_test(T, S, label)
    println("\n=== $label ===")
    x1_param  = KernelAbstractions.adapt(backend, zeros(T, N))
    u0_mat_T  = T.(u0_mat)

    model = Model(Ipopt.Optimizer)
    set_silent(model)

    set_optimizer_attribute(model, "tol", 1e-7)
    # set_optimizer_attribute(model, "constr_viol_tol", 1e-4)
    # set_optimizer_attribute(model, "acceptable_iter", 15)

    @variable(model, u[i=1:N],  start = x1_param[i])
    @objective(model, Min, sum((u[i] - x1_param[i])^2 for i in 1:N))
    @constraint(model, con[i=1:nx, s=1:n_samples], u[idx(i, 1, s)] == u0_mat_T[i, s])
    optimize!(model)
    sol = value.(u)
    sol_mat = reshape(sol, nx, nt, n_samples)

    viol = maximum(abs(sol_mat[i, 1, s] - u0_ic[i]) for i in 1:nx, s in 1:n_samples)

    cons_viol = norm(value.(con) .- u0_mat_T, Inf)

    # println("constr_viol_tol           = ", get_optimizer_attribute(model, "constr_viol_tol"))
    # println("acceptable_iter           = ", get_optimizer_attribute(model, "acceptable_iter"))

    println("  tol                                : ", get_optimizer_attribute(model, "tol"))
    println("  parameter eltype                   : ", T)
    println("  solution eltype                    : ", eltype(sol))
    println("  norm(con violation, Inf)           : ", cons_viol)
    println("  Externally computed viol           : ", viol)
    println("  constraints vs external match?     : ", abs(cons_viol - viol) < 1e-10)
end

run_test(Float32, Float64, "Float32 parameters → Float64 IPopt")
run_test(Float64, Float64, "Float64 parameters → Float64 IPopt")
run_test(Float32, Float32, "Float32 parameters → Float32 IPopt")
run_test(Float64, Float32, "Float64 parameters → Float32 IPopt")

println("\n-----------------------")