"""
Test whether passing Float64 parameters to ExaCore fixes the discrepancy
between MadNLP's reported constraint violation and the externally computed one.
"""

using ExaModels, MadNLP, MadNLPGPU
using CUDA, KernelAbstractions
using LinearAlgebra
backend = CUDABackend()

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

    core     = ExaCore(S, backend=backend)
    θ        = parameter(core, x1_param)
    u        = variable(core, 1:N; start = x1_param)
    objective(core, (u[i] - θ[i])^2 for i in 1:N)
    u0_param = parameter(core, u0_mat_T)
    constraint(core,
        (u[idx(i, 1, s)] - u0_param[i, s] for i in 1:nx, s in 1:n_samples);
        lcon = KernelAbstractions.adapt(backend, zeros(S, nx * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros(S, nx * n_samples)),
    )

    nlp    = ExaModel(core)
    # solver = MadNLP.MadNLPSolver(nlp; linear_solver=MadNLPGPU.CUDSSSolver, print_level=MadNLP.ERROR)
    solver = MadNLP.MadNLPSolver(nlp; linear_solver=MadNLPGPU.CUDSSSolver, print_level=MadNLP.ERROR, tol = 1e-7)
    println("get_tolerance = ", MadNLP.get_tolerance(eltype(nlp.meta.x0), typeof(solver.opt.kkt_system)))
    result = MadNLP.solve!(solver)
    sol    = Array(solution(result, u))

    sol_mat = reshape(sol, nx, nt, n_samples)
    viol = maximum(abs(sol_mat[i, 1, s] - u0_ic[i]) for i in 1:nx, s in 1:n_samples)

    cons_viol = norm(result.constraints, Inf)

    println("  parameter eltype                   : ", T)
    println("  solution eltype                    : ", eltype(solution(result, u)))
    println("  MadNLP primal_feas (cached)        : ", result.primal_feas)
    println("  norm(result.constraints, Inf)      : ", cons_viol)
    println("  Externally computed viol           : ", viol)
    println("  constraints vs external match?     : ", abs(cons_viol - viol) < 1e-10)
end

run_test(Float32, Float64, "Float32 parameters → Float64 ExaCore")
run_test(Float64, Float64, "Float64 parameters → Float64 ExaCore")
run_test(Float32, Float32, "Float32 parameters → Float32 ExaCore")
run_test(Float64, Float32, "Float64 parameters → Float32 ExaCore")

println("\n-----------------------")










T = Float32
S = Float32
label = "Float32 parameters → Float32 ExaCore"



println("\n=== $label ===")
x1_param  = KernelAbstractions.adapt(backend, zeros(T, N))
u0_mat_T  = T.(u0_mat)

core     = ExaCore(S, backend=backend) #This becomes Float64 as default. 
θ        = parameter(core, x1_param)
u        = variable(core, 1:N; start = x1_param)
objective(core, (u[i] - θ[i])^2 for i in 1:N)
u0_param = parameter(core, u0_mat_T)
constraint(core,
    (u[idx(i, 1, s)] - u0_param[i, s] for i in 1:nx, s in 1:n_samples);
    lcon = KernelAbstractions.adapt(backend, zeros(S, nx * n_samples)),
    ucon = KernelAbstractions.adapt(backend, zeros(S, nx * n_samples)),
)

nlp    = ExaModel(core) 
solver = MadNLP.MadNLPSolver(nlp; linear_solver=MadNLPGPU.CUDSSSolver, print_level=MadNLP.ERROR, 
        tol = 1e-10)
result = MadNLP.solve!(solver)
sol    = Array(solution(result, u))
solution(result, u)
sol_mat = reshape(sol, nx, nt, n_samples)
viol = maximum(abs(sol_mat[i, 1, s] - u0_ic[i]) for i in 1:nx, s in 1:n_samples)

cons_viol = norm(result.constraints, Inf)

println("  parameter eltype                   : ", T)
println("  solution eltype                    : ", eltype(solution(result, u)))
println("  MadNLP primal_feas (cached)        : ", result.primal_feas)
println("  norm(result.constraints, Inf)      : ", cons_viol)
println("  Externally computed viol           : ", viol)
println("  constraints vs external match?     : ", abs(cons_viol - viol) < 1e-10)

