using ExaModels


function heat_constraints!(core, u_flat, params)
    (; Nx, Nt, dx, u0, backend) = params

    idx(i, t) = (t-1)*Nx + i

    # --------------------------------------------------
    # 1. Initial condition: u(x,0) = u0(x)
    # --------------------------------------------------
    u0_data = [(i, u0[i]) for i in 1:Nx]
    constraint(core,
        (u_flat[d[1]] - d[2] for d in u0_data);
        lcon = KernelAbstractions.adapt(backend, zeros(Nx)),
        ucon = KernelAbstractions.adapt(backend, zeros(Nx))
    )

    # --------------------------------------------------
    # 2. Mass conservation: ∑ u(x,t) dx = 0 for all t
    # --------------------------------------------------
    constraint(core,
        (sum(u_flat[idx(i,t)] for i in 1:Nx) * dx for t in 1:Nt);
        lcon = KernelAbstractions.adapt(backend, zeros(Nt)),
        ucon = KernelAbstractions.adapt(backend, zeros(Nt))
    )

    return nothing
end

function ns_constraints!(core, u_flat, params)
    (; Nx, Ny, Nt, dx, dy, u0, backend) = params

    idx(i,j,t) = (t-1)*Nx*Ny + (j-1)*Nx + i

    # --------------------------------------------------
    # 1. Initial condition
    # --------------------------------------------------
    u0_data = [(idx(i,j,1), u0[i,j]) for i in 1:Nx for j in 1:Ny]
    constraint(core,
        (u_flat[d[1]] - d[2] for d in u0_data);
        lcon = KernelAbstractions.adapt(backend, zeros(Nx*Ny)),
        ucon = KernelAbstractions.adapt(backend, zeros(Nx*Ny))
    )

    # --------------------------------------------------
    # 2. Global mass conservation: ∑_{i,j} u(i,j,t)*dx*dy = M0 for t >= 2
    # --------------------------------------------------
    M0 = sum(u0[i,j] for i in 1:Nx for j in 1:Ny) * dx * dy
    constraint(core,
        (sum(u_flat[idx(i,j,t)] for i in 1:Nx for j in 1:Ny) * dx * dy - M0 for t in 2:Nt);
        lcon = KernelAbstractions.adapt(backend, zeros(Nt-1)),
        ucon = KernelAbstractions.adapt(backend, zeros(Nt-1))
    )

    return nothing
end

function rd_constraints!(core, u_flat, params)
    (; Nx, Nt, dx, dt, rho, nu, u0, backend) = params

    idx(i,t) = (t-1)*Nx + i

    # --------------------------------------------------
    # 1. Initial condition
    # --------------------------------------------------
    u0_data = [(i, u0[i]) for i in 1:Nx]
    constraint(core,
        (u_flat[d[1]] - d[2] for d in u0_data);
        lcon = KernelAbstractions.adapt(backend, zeros(Nx)),
        ucon = KernelAbstractions.adapt(backend, zeros(Nx))
    )

    # --------------------------------------------------
    # 2. Mass evolution (telescoping trapezoidal rule between adjacent steps):
    #    M[t] - M[t-1] = 0.5*dt*(S[t] + S[t-1]) + 0.5*dt*(F[t] + F[t-1])
    # where M[t] = ∑_i u[i,t]*dx
    #       S[t] = rho * ∑_i u[i,t]*(1-u[i,t])*dx
    #       F[t] = gL[t] - gR[t]  (4th-order FD boundary fluxes)
    # --------------------------------------------------
    constraint(core,
        (
            sum(u_flat[idx(i,t)]   for i in 1:Nx) * dx
            - sum(u_flat[idx(i,t-1)] for i in 1:Nx) * dx
            - 0.5*dt*rho*(
                sum(u_flat[idx(i,t)]   * (1 - u_flat[idx(i,t)])   for i in 1:Nx) * dx
                + sum(u_flat[idx(i,t-1)] * (1 - u_flat[idx(i,t-1)]) for i in 1:Nx) * dx
            )
            - 0.5*dt*(
                # F[t]: gL - gR at time t
                (
                    -nu*(-25*u_flat[idx(1,t)]  + 48*u_flat[idx(2,t)]  - 36*u_flat[idx(3,t)]  + 16*u_flat[idx(4,t)]  - 3*u_flat[idx(5,t)])  / (12*dx)
                  - -nu*( 25*u_flat[idx(Nx,t)] - 48*u_flat[idx(Nx-1,t)] + 36*u_flat[idx(Nx-2,t)] - 16*u_flat[idx(Nx-3,t)] + 3*u_flat[idx(Nx-4,t)]) / (12*dx)
                )
                # F[t-1]: gL - gR at time t-1
                + (
                    -nu*(-25*u_flat[idx(1,t-1)]  + 48*u_flat[idx(2,t-1)]  - 36*u_flat[idx(3,t-1)]  + 16*u_flat[idx(4,t-1)]  - 3*u_flat[idx(5,t-1)])  / (12*dx)
                  - -nu*( 25*u_flat[idx(Nx,t-1)] - 48*u_flat[idx(Nx-1,t-1)] + 36*u_flat[idx(Nx-2,t-1)] - 16*u_flat[idx(Nx-3,t-1)] + 3*u_flat[idx(Nx-4,t-1)]) / (12*dx)
                )
            )
            for t in 2:Nt
        );
        lcon = KernelAbstractions.adapt(backend, zeros(Nt-1)),
        ucon = KernelAbstractions.adapt(backend, zeros(Nt-1))
    )

    return nothing
end

function burgers_constraints!(core, u_flat, params)
    (; Nx, Nt, dx, dt, left_bc, backend) = params

    idx(i,t) = (t-1)*Nx + i

    # --------------------------------------------------
    # 1. Dirichlet BC at left boundary: u(1,t) = left_bc
    # --------------------------------------------------
    constraint(core,
        (u_flat[idx(1,t)] - left_bc for t in 1:Nt);
        lcon = KernelAbstractions.adapt(backend, zeros(Nt)),
        ucon = KernelAbstractions.adapt(backend, zeros(Nt))
    )

    # --------------------------------------------------
    # 2. Neumann BC at right boundary: u(Nx,t) = u(Nx-1,t)
    # --------------------------------------------------
    constraint(core,
        (u_flat[idx(Nx,t)] - u_flat[idx(Nx-1,t)] for t in 1:Nt);
        lcon = KernelAbstractions.adapt(backend, zeros(Nt)),
        ucon = KernelAbstractions.adapt(backend, zeros(Nt))
    )

    # --------------------------------------------------
    # 3. Mass evolution (telescoping trapezoidal rule):
    #    M[t] - M[t-1] = -0.5*dt*(F[t] + F[t-1])
    # where M[t] = ∑_i u[i,t]*dx,  F[t] = 0.5*u[Nx,t]^2 - 0.5*u[1,t]^2
    # --------------------------------------------------
    constraint(core,
        (
            sum(u_flat[idx(i,t)]   for i in 1:Nx) * dx
            - sum(u_flat[idx(i,t-1)] for i in 1:Nx) * dx
            + 0.5*dt*(
                (0.5*u_flat[idx(Nx,t)]^2   - 0.5*u_flat[idx(1,t)]^2)
                + (0.5*u_flat[idx(Nx,t-1)]^2 - 0.5*u_flat[idx(1,t-1)]^2)
            )
            for t in 2:Nt
        );
        lcon = KernelAbstractions.adapt(backend, zeros(Nt-1)),
        ucon = KernelAbstractions.adapt(backend, zeros(Nt-1))
    )

    return nothing
end
