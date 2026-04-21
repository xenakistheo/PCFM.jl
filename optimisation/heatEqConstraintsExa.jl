using MadNLP, MadNLPGPU
using CairoMakie
using BenchmarkTools
using ExaModels
using KernelAbstractions
using CUDA
using PCFM


backend = CPU()
# backend = CUDABackend()


CairoMakie.activate!()

include("plotUtils.jl")


Nx = 100
Nt = 100

# Suppose u1 is your data matrix in R^{Nx x Nt}
# Example:
u1 = randn(Nx, Nt)

L = 2π
dx = L / (Nx - 1)   # if grid includes both x=0 and x=2π
dt = 1/(Nt - 1)

X = collect(range(0, L; length=Nx))
T = collect(range(0, 1; length=Nt))



core = ExaCore(Float64; backend=backend)#CPU
#ExaModels operates on flat vectors 
idx(i,j) = i + (j-1)*Nx

N = Nx * Nt # Define total number of variables to solve for.
u = variable(core, 1:N, start = vec(u1))

# Embed (flat_index, data_value) pairs in the iteration set so ExaModels can
# access u1 values as constants without Julia array indexing on symbolic indices.
u1_data = [(k, vec(u1)[k]) for k in 1:N]

objective(core, (u[d[1]] - d[2])^2 for d in u1_data)

heat_constraints!(core, u, (Nx=Nx, Nt=Nt, dx=dx, u0=vec(u1[:,1]), backend=backend))

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
result = madnlp(nlp)
# @belapsed result = madnlp($nlp) # Solve optimisation using MadNLP
# typeof(result)
# fieldnames(result)
u_exa_vec = solution(result, u) # Extract shape. 
u_exa = Array(reshape(u_exa_vec, Nx, Nt))

# Solve and benchmark on GPU
@CUDA.time result = madnlp(nlp, linear_solver=MadNLPGPU.LapackCUDASolver)


##############
# Verify that result is same as that of JuMP
using JuMP

model = Model(MadNLP.Optimizer)

# Decision variables
@variable(model, u[1:Nx, 1:Nt])

for i in 1:Nx, j in 1:Nt
    set_start_value(u[i, j], u1[i, j])
end

@objective(model, Min, sum((u[i, j] - u1[i, j])^2 for i in 1:Nx, j in 1:Nt))
@constraint(model, [j in 1:Nt], dx * sum(u[i, j] for i in 1:(Nx-1)) == 0.0)

optimize!(model)

u_jump = value.(u)



# Compare the two solutions
u_jump == u_exa
isapprox(u_jump, u_exa)

plot_heatmap_evolution(X, T, u_jump, u_exa)

H1(u) = [dx*sum(u[i, j] for i in 1:(Nx-1)) for j in 1:Nt]
constraint_deviation_plot(u1, u_exa, T, H1; title="Exa")
constraint_deviation_plot(u1, u_jump, T, H1; title="Jump")
