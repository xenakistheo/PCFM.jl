using MadNLP, MadNLPGPU
using CairoMakie
using BenchmarkTools
using ExaModels
using KernelAbstractions
using CUDA
using PCFM


# backend = CPU()
backend = CUDABackend()


CairoMakie.activate!()

include("plotUtils.jl")


Nx = 100 # Points in Space
Nt = 100 # Points in Time
n_samples = 32 # Number of batches


# Suppose u1 is your data matrix in R^{Nx x Nt}
# Example:
u1 = randn(Nx, Nt, n_samples)

L = 2π
dx = L / (Nx - 1)   # if grid includes both x=0 and x=2π
dt = 1/(Nt - 1)

X = collect(range(0, L; length=Nx))
T = collect(range(0, 1; length=Nt))



core = ExaCore(Float64; backend=backend)

N = Nx * Nt * n_samples
u = variable(core, 1:N, start = vec(u1))

# flat index: k = i + (t-1)*Nx + (s-1)*Nx*Nt  (column-major, matches vec(u1))
u1_data = KernelAbstractions.adapt(backend, [(k, vec(u1)[k]) for k in 1:N])

objective(core, (u[d[1]] - d[2])^2 for d in u1_data)

heat_constraints!(core, u, (Nx=Nx, Nt=Nt, dx=dx, u0=u1[:, 1, :], n_samples=n_samples, backend=backend))

# constraint(core,
#     ((dx/dt) * sum(u[idx(i,j+1)]^2 - u[idx(i,j)]^2 for i in 1:(Nx-1))
#         for j in 1:(Nt-1)
#     );
#     lcon = KernelAbstractions.adapt(backend,zeros(Nt-1)), #lower constraints, on GPU
#     ucon = KernelAbstractions.adapt(backend,zeros(Nt - 1))
# )


# nlp = mean_zero_examodeL(u1, dx) # Defines optimisation problem

#Define optimisation problem
nlp = ExaModel(core)


# Solve and benchmark on CPU
@time result = madnlp(nlp)
# @belapsed result = madnlp($nlp) # Solve optimisation using MadNLP
# typeof(result)
# fieldnames(result)
@show MadNLP.Options().linear_solver 

u_exa_vec = solution(result, u)
u_exa = Array(reshape(u_exa_vec, Nx, Nt, n_samples))

# Solve and benchmark on GPU
result = madnlp(nlp, linear_solver=MadNLPGPU.LapackCUDASolver)
@time result = madnlp(nlp, linear_solver=MadNLPGPU.CUDSSSolver)

# Read about the difference between the two


##############
# Verify against JuMP on all samples
using JuMP

model = Model(MadNLP.Optimizer)
@variable(model, u_jmp[1:Nx, 1:Nt, 1:n_samples])
for i in 1:Nx, j in 1:Nt, s in 1:n_samples
    set_start_value(u_jmp[i, j, s], u1[i, j, s])
end
@objective(model, Min,
    sum((u_jmp[i,j,s] - u1[i,j,s])^2 for i in 1:Nx, j in 1:Nt, s in 1:n_samples))
@constraint(model, [i in 1:Nx, s in 1:n_samples], u_jmp[i, 1, s] == u1[i, 1, s])
@constraint(model, [j in 1:Nt, s in 1:n_samples],
    dx * sum(u_jmp[i,j,s] for i in 1:(Nx-1)) == 0.0)
@time optimize!(model)
u_jump = value.(u_jmp)   # (Nx, Nt, n_samples)

isapprox(u_jump, u_exa)

H1(u) = [dx*sum(u[i, j] for i in 1:(Nx-1)) for j in 1:Nt]
constraint_deviation_plot(u1[:,:,1], u_exa[:,:,1], T, H1; title="Exa sample 1")
constraint_deviation_plot(u1[:,:,1], u_jump[:,:,1], T, H1; title="Jump sample 1")
plot_heatmap_evolution(X, T, u_jump[:,:,1], u_exa[:,:,1])
