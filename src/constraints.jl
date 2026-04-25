using ExaModels
using KernelAbstractions

function heat_constraints!(model::Model, u, params)
    (; Nx, Nt, dx, u0, n_samples) = params
    """
    Constraints for the JuMP model of Heat Eq. 
    u should have shape (Nx, Nt, n_samples)
    """
    # @constraint(model, [i in 1:Nx, s in 1:n_samples], u[i, 1, s] == u0[i, 1, 1, s]) #IC
    @constraint(model, [j in 1:Nt, s in 1:n_samples], dx * sum(u[i,j,s] for i in 1:(Nx-1)) == 0.0) #Mass constraint 
end 



function heat_constraints!(core::ExaCore, u_flat, params)
    (; Nx, Nt, dx, u0, n_samples, backend) = params

    idx(i, t, s) = i + (t-1)*Nx + (s-1)*Nx*Nt

    u0_param = parameter(core, u0)

    # 1. Initial condition
    # constraint(
    #     core,
    #     (
    #         u_flat[idx(i, 1, s)] - u0_param[i, s]
    #         for i in 1:Nx, s in 1:n_samples
    #     );
    #     lcon = KernelAbstractions.adapt(backend, zeros(Nx * n_samples)),
    #     ucon = KernelAbstractions.adapt(backend, zeros(Nx * n_samples)),
    # )

    # 2. Mass conservation
    constraint(
        core,
        (
            sum(u_flat[idx(i, t, s)] for i in 1:Nx-1) * dx
            for t in 1:Nt, s in 1:n_samples
        );
        lcon = KernelAbstractions.adapt(backend, zeros(Nt * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros(Nt * n_samples)),
    )

    return nothing
end


function ns_constraints!(model::Model, u, params)
    (; Nx, Ny, Nt, dx, dy, u0, n_samples) = params
    # u has shape (Nx, Ny, Nt, n_samples)
    M0 = [sum(u0[i, j, s] for i in 1:Nx, j in 1:Ny) * dx * dy for s in 1:n_samples]

    # 1. Initial condition
    @constraint(model, [i in 1:Nx, j in 1:Ny, s in 1:n_samples], u[i, j, 1, s] == u0[i, j, s])

    # 2. Global mass conservation per sample for t >= 2
    @constraint(model, [t in 2:Nt, s in 1:n_samples],
        sum(u[i, j, t, s] for i in 1:Nx, j in 1:Ny) * dx * dy == M0[s])
end


function ns_constraints!(core::ExaCore, u_flat, params)
    (; Nx, Ny, Nt, dx, dy, u0, n_samples, backend) = params

    # u0 is (Nx, Ny, n_samples)
    # flat index: i + (j-1)*Nx + (t-1)*Nx*Ny + (s-1)*Nx*Ny*Nt
    idx(i, j, t, s) = i + (j-1)*Nx + (t-1)*Nx*Ny + (s-1)*Nx*Ny*Nt

    # --------------------------------------------------
    # 1. Initial condition for each sample
    # --------------------------------------------------
    u0_data = [(idx(i, j, 1, s), u0[i, j, s]) for i in 1:Nx for j in 1:Ny for s in 1:n_samples]
    constraint(core,
        (u_flat[d[1]] - d[2] for d in u0_data);
        lcon = KernelAbstractions.adapt(backend, zeros(Nx * Ny * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros(Nx * Ny * n_samples))
    )

    # --------------------------------------------------
    # 2. Global mass conservation per sample: ∑_{i,j} u*dx*dy = M0[s] for t >= 2
    # Embed (t, s, M0[s]) as data so ExaModels can access M0 as a constant
    # --------------------------------------------------
    M0 = [sum(u0[i, j, s] for i in 1:Nx for j in 1:Ny) * dx * dy for s in 1:n_samples]
    ts_M0 = [(t, s, M0[s]) for t in 2:Nt for s in 1:n_samples]
    constraint(core,
        (sum(u_flat[idx(i, j, d[1], d[2])] for i in 1:Nx for j in 1:Ny) * dx * dy - d[3]
            for d in ts_M0);
        lcon = KernelAbstractions.adapt(backend, zeros((Nt-1) * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros((Nt-1) * n_samples))
    )

    return nothing
end

function rd_constraints!(model::Model, u, params)
    (; Nx, Nt, dx, dt, rho, nu, u0, n_samples) = params
    # u has shape (Nx, Nt, n_samples)

    # 1. Initial condition
    @constraint(model, [i in 1:Nx, s in 1:n_samples], u[i, 1, s] == u0[i, s])

    # 2. Mass evolution (trapezoidal rule)
    @constraint(model, [t in 2:Nt, s in 1:n_samples],
        sum(u[i, t, s] for i in 1:Nx) * dx
        - sum(u[i, t-1, s] for i in 1:Nx) * dx
        - 0.5*dt*rho*(
            sum(u[i, t,   s] * (1 - u[i, t,   s]) for i in 1:Nx) * dx
            + sum(u[i, t-1, s] * (1 - u[i, t-1, s]) for i in 1:Nx) * dx
        )
        - 0.5*dt*(
            (
                -nu*(-25*u[1,    t,   s] + 48*u[2,    t,   s] - 36*u[3,    t,   s] + 16*u[4,    t,   s] - 3*u[5,    t,   s]) / (12*dx)
              - -nu*( 25*u[Nx,   t,   s] - 48*u[Nx-1, t,   s] + 36*u[Nx-2, t,   s] - 16*u[Nx-3, t,   s] + 3*u[Nx-4, t,   s]) / (12*dx)
            )
            + (
                -nu*(-25*u[1,    t-1, s] + 48*u[2,    t-1, s] - 36*u[3,    t-1, s] + 16*u[4,    t-1, s] - 3*u[5,    t-1, s]) / (12*dx)
              - -nu*( 25*u[Nx,   t-1, s] - 48*u[Nx-1, t-1, s] + 36*u[Nx-2, t-1, s] - 16*u[Nx-3, t-1, s] + 3*u[Nx-4, t-1, s]) / (12*dx)
            )
        ) == 0.0)
end


function rd_constraints!(core::ExaCore, u_flat, params)
    (; Nx, Nt, dx, dt, rho, nu, u0, n_samples, backend) = params

    # u0 is (Nx, n_samples)
    # flat index: i + (t-1)*Nx + (s-1)*Nx*Nt
    idx(i, t, s) = i + (t-1)*Nx + (s-1)*Nx*Nt

    # --------------------------------------------------
    # 1. Initial condition for each sample
    # --------------------------------------------------
    u0_data = [(idx(i, 1, s), u0[i, s]) for i in 1:Nx for s in 1:n_samples]
    constraint(core,
        (u_flat[d[1]] - d[2] for d in u0_data);
        lcon = KernelAbstractions.adapt(backend, zeros(Nx * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros(Nx * n_samples))
    )

    # --------------------------------------------------
    # 2. Mass evolution (telescoping trapezoidal rule) for all samples:
    #    M[t,s] - M[t-1,s] = 0.5*dt*(S[t,s] + S[t-1,s]) + 0.5*dt*(F[t,s] + F[t-1,s])
    # --------------------------------------------------
    ts_pairs = [(t, s) for t in 2:Nt for s in 1:n_samples]
    constraint(core,
        (
            sum(u_flat[idx(i, d[1],   d[2])] for i in 1:Nx) * dx
            - sum(u_flat[idx(i, d[1]-1, d[2])] for i in 1:Nx) * dx
            - 0.5*dt*rho*(
                sum(u_flat[idx(i, d[1],   d[2])] * (1 - u_flat[idx(i, d[1],   d[2])]) for i in 1:Nx) * dx
                + sum(u_flat[idx(i, d[1]-1, d[2])] * (1 - u_flat[idx(i, d[1]-1, d[2])]) for i in 1:Nx) * dx
            )
            - 0.5*dt*(
                (
                    -nu*(-25*u_flat[idx(1,    d[1],   d[2])] + 48*u_flat[idx(2,    d[1],   d[2])] - 36*u_flat[idx(3,    d[1],   d[2])] + 16*u_flat[idx(4,    d[1],   d[2])] - 3*u_flat[idx(5,    d[1],   d[2])]) / (12*dx)
                  - -nu*( 25*u_flat[idx(Nx,   d[1],   d[2])] - 48*u_flat[idx(Nx-1, d[1],   d[2])] + 36*u_flat[idx(Nx-2, d[1],   d[2])] - 16*u_flat[idx(Nx-3, d[1],   d[2])] + 3*u_flat[idx(Nx-4, d[1],   d[2])]) / (12*dx)
                )
                + (
                    -nu*(-25*u_flat[idx(1,    d[1]-1, d[2])] + 48*u_flat[idx(2,    d[1]-1, d[2])] - 36*u_flat[idx(3,    d[1]-1, d[2])] + 16*u_flat[idx(4,    d[1]-1, d[2])] - 3*u_flat[idx(5,    d[1]-1, d[2])]) / (12*dx)
                  - -nu*( 25*u_flat[idx(Nx,   d[1]-1, d[2])] - 48*u_flat[idx(Nx-1, d[1]-1, d[2])] + 36*u_flat[idx(Nx-2, d[1]-1, d[2])] - 16*u_flat[idx(Nx-3, d[1]-1, d[2])] + 3*u_flat[idx(Nx-4, d[1]-1, d[2])]) / (12*dx)
                )
            )
            for d in ts_pairs
        );
        lcon = KernelAbstractions.adapt(backend, zeros((Nt-1) * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros((Nt-1) * n_samples))
    )

    return nothing
end

function burgers_constraints!(model::Model, u, params)
    (; Nx, Nt, dx, dt, left_bc, n_samples) = params
    # u has shape (Nx, Nt, n_samples)

    # 1. Dirichlet BC at left boundary
    @constraint(model, [t in 1:Nt, s in 1:n_samples], u[1, t, s] == left_bc)

    # 2. Neumann BC at right boundary
    @constraint(model, [t in 1:Nt, s in 1:n_samples], u[Nx, t, s] == u[Nx-1, t, s])

    # 3. Mass evolution (trapezoidal rule)
    @constraint(model, [t in 2:Nt, s in 1:n_samples],
        sum(u[i, t,   s] for i in 1:Nx) * dx
        - sum(u[i, t-1, s] for i in 1:Nx) * dx
        + 0.5*dt*(
            (0.5*u[Nx, t,   s]^2 - 0.5*u[1, t,   s]^2)
            + (0.5*u[Nx, t-1, s]^2 - 0.5*u[1, t-1, s]^2)
        ) == 0.0)
end


function burgers_constraints!(core::ExaCore, u_flat, params)
    (; Nx, Nt, dx, dt, left_bc, n_samples, backend) = params

    # flat index: i + (t-1)*Nx + (s-1)*Nx*Nt
    idx(i, t, s) = i + (t-1)*Nx + (s-1)*Nx*Nt

    # --------------------------------------------------
    # 1. Dirichlet BC at left boundary: u(1,t,s) = left_bc
    # --------------------------------------------------
    ts_pairs_all = [(t, s) for t in 1:Nt for s in 1:n_samples]
    constraint(core,
        (u_flat[idx(1, d[1], d[2])] - left_bc for d in ts_pairs_all);
        lcon = KernelAbstractions.adapt(backend, zeros(Nt * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros(Nt * n_samples))
    )

    # --------------------------------------------------
    # 2. Neumann BC at right boundary: u(Nx,t,s) = u(Nx-1,t,s)
    # --------------------------------------------------
    constraint(core,
        (u_flat[idx(Nx, d[1], d[2])] - u_flat[idx(Nx-1, d[1], d[2])] for d in ts_pairs_all);
        lcon = KernelAbstractions.adapt(backend, zeros(Nt * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros(Nt * n_samples))
    )

    # --------------------------------------------------
    # 3. Mass evolution (telescoping trapezoidal rule) for all samples:
    #    M[t,s] - M[t-1,s] = -0.5*dt*(F[t,s] + F[t-1,s])
    # --------------------------------------------------
    ts_pairs_inner = [(t, s) for t in 2:Nt for s in 1:n_samples]
    constraint(core,
        (
            sum(u_flat[idx(i, d[1],   d[2])] for i in 1:Nx) * dx
            - sum(u_flat[idx(i, d[1]-1, d[2])] for i in 1:Nx) * dx
            + 0.5*dt*(
                (0.5*u_flat[idx(Nx, d[1],   d[2])]^2 - 0.5*u_flat[idx(1, d[1],   d[2])]^2)
                + (0.5*u_flat[idx(Nx, d[1]-1, d[2])]^2 - 0.5*u_flat[idx(1, d[1]-1, d[2])]^2)
            )
            for d in ts_pairs_inner
        );
        lcon = KernelAbstractions.adapt(backend, zeros((Nt-1) * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros((Nt-1) * n_samples))
    )

    return nothing
end
