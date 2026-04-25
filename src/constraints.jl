using ExaModels
using KernelAbstractions

function heat_constraints!(model::Model, u, u0, nt, n_samples, grid_points, grid_spacing, dt, params=nothing)
    """
    Constraints for the JuMP model of Heat Eq. 
    u should have shape (nx, nt, n_samples)
    """
    nx  = grid_points[1]                                                                                                                                             
    dx = grid_spacing[1]
    @constraint(model, [i in 1:nx, s in 1:n_samples], u[i, 1, s] == u0[i, 1, 1, s]) #IC
    @constraint(model, [j in 2:nt, s in 1:n_samples], sum(u[i,j,s] for i in 1:(nx-1)) == sum(u0[i,1,1,s] for i in 1:(nx-1))) #Mass constraint 
end 



function heat_constraints!(core::ExaCore, u_flat, params)
    (; nx, nt, dx, u0, n_samples, backend) = params

    idx(i, t, s) = i + (t-1)*nx + (s-1)*nx*nt

    u0_param = parameter(core, u0)

    # 1. Initial condition
    constraint(
        core,
        (
            u_flat[idx(i, 1, s)] - u0_param[i, s]
            for i in 1:nx, s in 1:n_samples
        );
        lcon = KernelAbstractions.adapt(backend, zeros(nx * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros(nx * n_samples)),
    )

    # 2. Mass conservation
    constraint(
        core,
        (
            sum(u_flat[idx(i, t, s)] for i in 1:nx-1) - sum(u0_param[i, s] for i in 1:nx-1)
            for t in 2:nt, s in 1:n_samples
        );
        lcon = KernelAbstractions.adapt(backend, zeros(nt * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros(nt * n_samples)),
    )

    return nothing
end


function ns_constraints!(model::Model, u, u0, nt, n_samples, grid_points, grid_spacing, dt, params=nothing)
    nx, ny  = grid_points                                                                                                                                            
    dx, dy = grid_spacing

    # u has shape (nx, ny, nt, n_samples)
    M0 = [sum(u0[i, j, s] for i in 1:nx, j in 1:ny) for s in 1:n_samples]

    # 1. Initial condition
    @constraint(model, [i in 1:nx, j in 1:ny, s in 1:n_samples], u[i, j, 1, s] == u0[i, j, s])

    # 2. Global mass conservation per sample for t >= 2
    @constraint(model, [t in 2:nt, s in 1:n_samples],
        sum(u[i, j, t, s] for i in 1:nx, j in 1:ny) == M0[s])
end


function ns_constraints!(core::ExaCore, u_flat, params)
    (; nx, ny, nt, dx, dy, u0, n_samples, backend) = params

    # u0 is (nx, ny, n_samples)
    # flat index: i + (j-1)*nx + (t-1)*nx*ny + (s-1)*nx*ny*nt
    idx(i, j, t, s) = i + (j-1)*nx + (t-1)*nx*ny + (s-1)*nx*ny*nt

    # --------------------------------------------------
    # 1. Initial condition for each sample
    # --------------------------------------------------
    u0_data = [(idx(i, j, 1, s), u0[i, j, s]) for i in 1:nx for j in 1:Ny for s in 1:n_samples]
    constraint(core,
        (u_flat[d[1]] - d[2] for d in u0_data);
        lcon = KernelAbstractions.adapt(backend, zeros(nx * ny * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros(nx * ny * n_samples))
    )

    # --------------------------------------------------
    # 2. Global mass conservation per sample: ∑_{i,j} u*dx*dy = M0[s] for t >= 2
    # Embed (t, s, M0[s]) as data so ExaModels can access M0 as a constant
    # --------------------------------------------------
    M0 = [sum(u0[i, j, s] for i in 1:nx for j in 1:ny) for s in 1:n_samples]
    ts_M0 = [(t, s, M0[s]) for t in 2:nt for s in 1:n_samples]
    constraint(core,
        (sum(u_flat[idx(i, j, d[1], d[2])] for i in 1:nx for j in 1:ny) - d[3]
            for d in ts_M0);
        lcon = KernelAbstractions.adapt(backend, zeros((nt-1) * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros((nt-1) * n_samples))
    )

    return nothing
end

function rd_constraints!(model::Model, u, u0, nt, n_samples, grid_points, grid_spacing, dt, params=[;])
    nx  = grid_points[1]                                                                                                                                             
    dx = grid_spacing[1]
    rho = get(params, :rho, 1.0) # Default values                                                                                                                                      
    nu  = get(params, :nu, 0.01)

    # u has shape (nx, nt, n_samples)

    # 1. Initial condition
    @constraint(model, [i in 1:nx, s in 1:n_samples], u[i, 1, s] == u0[i, s])

    # 2. Mass evolution (trapezoidal rule)
    @constraint(model, [t in 2:nt, s in 1:n_samples],
        sum(u[i, t, s] for i in 1:nx) * dx
        - sum(u[i, t-1, s] for i in 1:nx) * dx
        - 0.5*dt*rho*(
            sum(u[i, t,   s] * (1 - u[i, t,   s]) for i in 1:nx) * dx
            + sum(u[i, t-1, s] * (1 - u[i, t-1, s]) for i in 1:nx) * dx
        )
        - 0.5*dt*(
            (
                -nu*(-25*u[1,    t,   s] + 48*u[2,    t,   s] - 36*u[3,    t,   s] + 16*u[4,    t,   s] - 3*u[5,    t,   s]) / (12*dx)
              - -nu*( 25*u[nx,   t,   s] - 48*u[nx-1, t,   s] + 36*u[nx-2, t,   s] - 16*u[nx-3, t,   s] + 3*u[nx-4, t,   s]) / (12*dx)
            )
            + (
                -nu*(-25*u[1,    t-1, s] + 48*u[2,    t-1, s] - 36*u[3,    t-1, s] + 16*u[4,    t-1, s] - 3*u[5,    t-1, s]) / (12*dx)
              - -nu*( 25*u[nx,   t-1, s] - 48*u[nx-1, t-1, s] + 36*u[nx-2, t-1, s] - 16*u[nx-3, t-1, s] + 3*u[nx-4, t-1, s]) / (12*dx)
            )
        ) == 0.0)
end


function rd_constraints!(core::ExaCore, u_flat, params)
    (; nx, nt, dx, dt, rho, nu, u0, n_samples, backend) = params

    # u0 is (nx, n_samples)
    # flat index: i + (t-1)*nx + (s-1)*nx*nt
    idx(i, t, s) = i + (t-1)*nx + (s-1)*nx*nt

    # --------------------------------------------------
    # 1. Initial condition for each sample
    # --------------------------------------------------
    u0_data = [(idx(i, 1, s), u0[i, s]) for i in 1:nx for s in 1:n_samples]
    constraint(core,
        (u_flat[d[1]] - d[2] for d in u0_data);
        lcon = KernelAbstractions.adapt(backend, zeros(nx * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros(nx * n_samples))
    )

    # --------------------------------------------------
    # 2. Mass evolution (telescoping trapezoidal rule) for all samples:
    #    M[t,s] - M[t-1,s] = 0.5*dt*(S[t,s] + S[t-1,s]) + 0.5*dt*(F[t,s] + F[t-1,s])
    # --------------------------------------------------
    ts_pairs = [(t, s) for t in 2:nt for s in 1:n_samples]
    constraint(core,
        (
            sum(u_flat[idx(i, d[1],   d[2])] for i in 1:nx) * dx
            - sum(u_flat[idx(i, d[1]-1, d[2])] for i in 1:nx) * dx
            - 0.5*dt*rho*(
                sum(u_flat[idx(i, d[1],   d[2])] * (1 - u_flat[idx(i, d[1],   d[2])]) for i in 1:nx) * dx
                + sum(u_flat[idx(i, d[1]-1, d[2])] * (1 - u_flat[idx(i, d[1]-1, d[2])]) for i in 1:nx) * dx
            )
            - 0.5*dt*(
                (
                    -nu*(-25*u_flat[idx(1,    d[1],   d[2])] + 48*u_flat[idx(2,    d[1],   d[2])] - 36*u_flat[idx(3,    d[1],   d[2])] + 16*u_flat[idx(4,    d[1],   d[2])] - 3*u_flat[idx(5,    d[1],   d[2])]) / (12*dx)
                  - -nu*( 25*u_flat[idx(nx,   d[1],   d[2])] - 48*u_flat[idx(nx-1, d[1],   d[2])] + 36*u_flat[idx(nx-2, d[1],   d[2])] - 16*u_flat[idx(Nx-3, d[1],   d[2])] + 3*u_flat[idx(nx-4, d[1],   d[2])]) / (12*dx)
                )
                + (
                    -nu*(-25*u_flat[idx(1,    d[1]-1, d[2])] + 48*u_flat[idx(2,    d[1]-1, d[2])] - 36*u_flat[idx(3,    d[1]-1, d[2])] + 16*u_flat[idx(4,    d[1]-1, d[2])] - 3*u_flat[idx(5,    d[1]-1, d[2])]) / (12*dx)
                  - -nu*( 25*u_flat[idx(nx,   d[1]-1, d[2])] - 48*u_flat[idx(nx-1, d[1]-1, d[2])] + 36*u_flat[idx(nx-2, d[1]-1, d[2])] - 16*u_flat[idx(nx-3, d[1]-1, d[2])] + 3*u_flat[idx(nx-4, d[1]-1, d[2])]) / (12*dx)
                )
            )
            for d in ts_pairs
        );
        lcon = KernelAbstractions.adapt(backend, zeros((nt-1) * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros((nt-1) * n_samples))
    )

    return nothing
end

function burgers_constraints!(model::Model, u, u0, nt, n_samples, grid_points, grid_spacing, dt, params=nothing)
    nx  = grid_points[1]                                                                                                                                             
    dx = grid_spacing[1]

    # u has shape (nx, nt, n_samples)

    # 1. Dirichlet BC at left boundary
    @constraint(model, [t in 1:nt, s in 1:n_samples], u[1, t, s] == left_bc)

    # 2. Neumann BC at right boundary
    @constraint(model, [t in 1:nt, s in 1:n_samples], u[nx, t, s] == u[nx-1, t, s])

    # 3. Mass evolution (trapezoidal rule)
    @constraint(model, [t in 2:nt, s in 1:n_samples],
        sum(u[i, t,   s] for i in 1:nx) * dx
        - sum(u[i, t-1, s] for i in 1:nx) * dx
        + 0.5*dt*(
            (0.5*u[nx, t,   s]^2 - 0.5*u[1, t,   s]^2)
            + (0.5*u[nx, t-1, s]^2 - 0.5*u[1, t-1, s]^2)
        ) == 0.0)
end


function burgers_constraints!(core::ExaCore, u_flat, params)
    (; nx, Nt, dx, dt, left_bc, n_samples, backend) = params

    # flat index: i + (t-1)*nx + (s-1)*nx*nt
    idx(i, t, s) = i + (t-1)*nx + (s-1)*nx*nt

    # --------------------------------------------------
    # 1. Dirichlet BC at left boundary: u(1,t,s) = left_bc
    # --------------------------------------------------
    ts_pairs_all = [(t, s) for t in 1:nt for s in 1:n_samples]
    constraint(core,
        (u_flat[idx(1, d[1], d[2])] - left_bc for d in ts_pairs_all);
        lcon = KernelAbstractions.adapt(backend, zeros(nt * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros(nt * n_samples))
    )

    # --------------------------------------------------
    # 2. Neumann BC at right boundary: u(nx,t,s) = u(nx-1,t,s)
    # --------------------------------------------------
    constraint(core,
        (u_flat[idx(nx, d[1], d[2])] - u_flat[idx(nx-1, d[1], d[2])] for d in ts_pairs_all);
        lcon = KernelAbstractions.adapt(backend, zeros(nt * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros(nt * n_samples))
    )

    # --------------------------------------------------
    # 3. Mass evolution (telescoping trapezoidal rule) for all samples:
    #    M[t,s] - M[t-1,s] = -0.5*dt*(F[t,s] + F[t-1,s])
    # --------------------------------------------------
    ts_pairs_inner = [(t, s) for t in 2:nt for s in 1:n_samples]
    constraint(core,
        (
            sum(u_flat[idx(i, d[1],   d[2])] for i in 1:nx) * dx
            - sum(u_flat[idx(i, d[1]-1, d[2])] for i in 1:nx) * dx
            + 0.5*dt*(
                (0.5*u_flat[idx(nx, d[1],   d[2])]^2 - 0.5*u_flat[idx(1, d[1],   d[2])]^2)
                + (0.5*u_flat[idx(nx, d[1]-1, d[2])]^2 - 0.5*u_flat[idx(1, d[1]-1, d[2])]^2)
            )
            for d in ts_pairs_inner
        );
        lcon = KernelAbstractions.adapt(backend, zeros((nt-1) * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros((nt-1) * n_samples))
    )

    return nothing
end
