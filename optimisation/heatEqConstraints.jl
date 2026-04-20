using JuMP, MadNLP, CairoMakie
using BenchmarkTools

CairoMakie.activate!()

include("plotUtils.jl")

# Problem sizes
Nx = 100
Nt = 100

# Suppose u1 is your data matrix in R^{Nx x Nt}
# Example:
u1 = randn(Nx, Nt)

# Choose the correct dx for your grid convention:
L = 2π
dx = L / (Nx - 1)   # if grid includes both x=0 and x=2π
dt = 1/(Nt - 1)

X = collect(range(0, L; length=Nx))
T = collect(range(0, 1; length=Nt))



######################################
# Linear Constraint
model = Model(MadNLP.Optimizer)

# Decision variables
@variable(model, u[1:Nx, 1:Nt])

# Good practice: initialize at the data
for i in 1:Nx, j in 1:Nt
    set_start_value(u[i, j], u1[i, j])
end

# Objective: minimize ||u - u1||_F^2
@objective(model, Min, sum((u[i, j] - u1[i, j])^2 for i in 1:Nx, j in 1:Nt))

# Mean-zero constraint for each time slice
@constraint(model, [j in 1:Nt], dx * sum(u[i, j] for i in 1:(Nx-1)) == 0.0)

optimize!(model)

u_proj = value.(u)

@assert size(u1) == size(u_proj)




plot_heatmap_evolution(X, T, u1, u_proj, "Linear Constraint")

# plot_evolution(u1, u_proj, Nt)

H1(u) = [dx*sum(u[i, j] for i in 1:(Nx-1)) for j in 1:Nt]
constraint_deviation_plot(u1, u_proj, T, H1)



######################################
# Nonlinear Conservation Law
model2 = Model(MadNLP.Optimizer)
@variable(model2, u[1:Nx, 1:Nt])

for i in 1:Nx, j in 1:Nt
    set_start_value(u[i, j], u1[i, j])
end

@objective(model2, Min, sum((u[i, j] - u1[i, j])^2 for i in 1:Nx, j in 1:Nt))

H2(u, j) = (dx/dt) * sum((u[i, j+1]^2 - u[i, j]^2) for i in 1:(Nx-1))
@constraint(model2, [j in 1:Nt-1], H2(u,j) == 0.0)

optimize!(model2)

u_proj2 = value.(u)

H2(u) = [H2(u,j) for j in 1:Nt-1]
constraint_deviation_plot(u1, u_proj2, T, H2)
plot_heatmap_evolution(X, T, u1, u_proj2, "Nonlinear Conservation Law")


######################################
# Neumann (Flux) BC
model3 = Model(MadNLP.Optimizer)
@variable(model3, u[1:Nx, 1:Nt])

for i in 1:Nx, j in 1:Nt
    set_start_value(u[i, j], u1[i, j])
end

@objective(model3, Min, sum((u[i, j] - u1[i, j])^2 for i in 1:Nx, j in 1:Nt))

H2(u, j) = (dx/dt) * sum((u[i, j+1]^2 - u[i, j]^2) for i in 1:(Nx-1))
# @constraint(model2, [j in 1:Nt-1], H2(u,j) >= 0.0)
@constraint(model3, [j in 1:Nt-1], (u[2,j] - u[1,j])/dx + sin(T[j]) == 0.0)
@constraint(model3, [j in 1:Nt-1], (u[end,j] - u[end-1,j])/dx - sin(T[j]) == 0.0)

optimize!(model3)

u_proj3 = value.(u)


plot_heatmap_evolution(X, T, u1, u_proj3, "Neumann (Flux) BC")

