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



function heat_constraints!(core::ExaCore, u_flat, u0_flat, nt, n_samples, grid_points, grid_spacing, dt, params=nothing; backend=CPU())
    nx  = grid_points[1]                                                                                                                                             
    dx = grid_spacing[1]

    idx(i, t, s) = i + (t-1)*nx + (s-1)*nx*nt

    u0_param = parameter(core, u0_flat)

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
        lcon = KernelAbstractions.adapt(backend, zeros((nt-1) * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros((nt-1) * n_samples)),
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


function ns_constraints!(core::ExaCore, u_flat, u0_flat, nt, n_samples, grid_points, grid_spacing, dt, params=nothing; backend=CPU())
    nx, ny  = grid_points                                                                                                                                            
    dx, dy = grid_spacing

    # u0 is (nx, ny, n_samples)
    # flat index: i + (j-1)*nx + (t-1)*nx*ny + (s-1)*nx*ny*nt
    idx(i, j, t, s) = i + (j-1)*nx + (t-1)*nx*ny + (s-1)*nx*ny*nt

    # --------------------------------------------------
    # 1. Initial condition for each sample
    # --------------------------------------------------
    u0_param = parameter(core, u0_flat)

    constraint(core,
        (u_flat[idx(i, j, 1, s)] - u0_param[i, j, s]
         for i in 1:nx, j in 1:ny, s in 1:n_samples);
        lcon = KernelAbstractions.adapt(backend, zeros(nx * ny * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros(nx * ny * n_samples))
    )

    # --------------------------------------------------
    # 2. Global mass conservation per sample: ∑_{i,j} u = M0[s] for t >= 2
    # --------------------------------------------------
    M0 = dropdims(sum(u0_flat, dims=(1, 2)), dims=(1, 2))  # shape: (n_samples,)
    M0_param = parameter(core, M0)

    ts_pairs = [(t, s) for t in 2:nt for s in 1:n_samples]
    constraint(core,
        (sum(u_flat[idx(i, j, d[1], d[2])] for i in 1:nx, j in 1:ny) - M0_param[d[2]]
            for d in ts_pairs);
        lcon = KernelAbstractions.adapt(backend, zeros((nt-1) * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros((nt-1) * n_samples))
    )

    return nothing
end

function rd_constraints!(model::Model, u, u0, nt, n_samples, grid_points, grid_spacing, dt, params=(;))
    nx  = grid_points[1]                                                                                                                                             
    dx = grid_spacing[1]
    rho = get(params, :rho, 0.01) # Default values                                                                                                                                      
    nu  = get(params, :nu, 0.005)

    # u has shape (nx, nt, n_samples)

    # 1. Initial condition
    @constraint(model, [i in 1:nx, s in 1:n_samples], u[i, 1, s] == u0[i, 1, 1, s])

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


function rd_constraints!(core::ExaCore, u_flat, u0_flat, nt, n_samples, grid_points, grid_spacing, dt, params=(;); backend=CPU())
    nx  = grid_points[1]                                                                                                                                             
    dx = grid_spacing[1]
    rho = get(params, :rho, 0.01) # Default values                                                                                                                                      
    nu  = get(params, :nu, 0.005)

    # u0 is (nx, n_samples)
    # flat index: i + (t-1)*nx + (s-1)*nx*nt
    idx(i, t, s) = i + (t-1)*nx + (s-1)*nx*nt

    # --------------------------------------------------
    # 1. Initial condition for each sample
    # --------------------------------------------------
    u0_param = parameter(core, u0_flat)

    constraint(core,
        (u_flat[idx(i, 1, s)] - u0_param[i, s]
         for i in 1:nx, s in 1:n_samples);
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
                  - -nu*( 25*u_flat[idx(nx,   d[1],   d[2])] - 48*u_flat[idx(nx-1, d[1],   d[2])] + 36*u_flat[idx(nx-2, d[1],   d[2])] - 16*u_flat[idx(nx-3, d[1],   d[2])] + 3*u_flat[idx(nx-4, d[1],   d[2])]) / (12*dx)
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

function rd_constraints_2!(model::Model, u, u0, nt, n_samples, grid_points, grid_spacing, dt, params=(;))
    nx  = grid_points[1]
    dx = grid_spacing[1]
    rho = get(params, :rho, 0.01)
    nu  = get(params, :nu, 0.005)

    # 1. Initial condition
    @constraint(model, [i in 1:nx, s in 1:n_samples], u[i, 1, s] == u0[i, 1, 1, s])

    # 2. Mass evolution (trapezoidal rule, 1st-order boundary flux)
    @constraint(model, [t in 2:nt, s in 1:n_samples],
        sum(u[i, t, s] for i in 1:nx) * dx
        - sum(u[i, t-1, s] for i in 1:nx) * dx
        - 0.5*dt*rho*(
            sum(u[i, t,   s] * (1 - u[i, t,   s]) for i in 1:nx) * dx
            + sum(u[i, t-1, s] * (1 - u[i, t-1, s]) for i in 1:nx) * dx
        )
        - 0.5*dt*(
            (
                -nu*(u[2,    t,   s] - u[1,  t,   s]) / dx
              - -nu*(u[nx,   t,   s] - u[nx-1, t,   s]) / dx
            )
            + (
                -nu*(u[2,    t-1, s] - u[1,  t-1, s]) / dx
              - -nu*(u[nx,   t-1, s] - u[nx-1, t-1, s]) / dx
            )
        ) == 0.0)
end


function rd_constraints_2!(core::ExaCore, u_flat, u0_flat, nt, n_samples, grid_points, grid_spacing, dt, params=(;); backend=CPU())
    nx  = grid_points[1]
    dx = grid_spacing[1]
    rho = get(params, :rho, 0.01)
    nu  = get(params, :nu, 0.005)

    idx(i, t, s) = i + (t-1)*nx + (s-1)*nx*nt

    # 1. Initial condition
    u0_param = parameter(core, u0_flat)

    constraint(core,
        (u_flat[idx(i, 1, s)] - u0_param[i, s]
         for i in 1:nx, s in 1:n_samples);
        lcon = KernelAbstractions.adapt(backend, zeros(nx * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros(nx * n_samples))
    )

    # 2. Mass evolution (trapezoidal rule, 1st-order boundary flux)
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
                    -nu*(u_flat[idx(2,  d[1],   d[2])] - u_flat[idx(1,    d[1],   d[2])]) / dx
                  - -nu*(u_flat[idx(nx, d[1],   d[2])] - u_flat[idx(nx-1, d[1],   d[2])]) / dx
                )
                + (
                    -nu*(u_flat[idx(2,  d[1]-1, d[2])] - u_flat[idx(1,    d[1]-1, d[2])]) / dx
                  - -nu*(u_flat[idx(nx, d[1]-1, d[2])] - u_flat[idx(nx-1, d[1]-1, d[2])]) / dx
                )
            )
            for d in ts_pairs
        );
        lcon = KernelAbstractions.adapt(backend, zeros((nt-1) * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros((nt-1) * n_samples))
    )

    return nothing
end


function burgers_constraints_BC_Mass!(model::Model, u, u0, nt, n_samples, grid_points, grid_spacing, dt, params=(;))
    nx  = grid_points[1]
    dx = grid_spacing[1]
    left_bc_vec = get(params, :left_bc, zeros(Float32, n_samples))

    # u has shape (nx, nt, n_samples)

    # 1. Dirichlet BC at left boundary
    @constraint(model, [t in 1:nt, s in 1:n_samples], u[1, t, s] == left_bc_vec[s])

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


function burgers_constraints_BC_Mass!(core::ExaCore, u_flat, u0_flat, nt, n_samples, grid_points, grid_spacing, dt, params=(;); backend=CPU())
    nx  = grid_points[1]
    dx = grid_spacing[1]
    left_bc_vec = get(params, :left_bc, zeros(Float32, n_samples))
    left_bc_param = parameter(core, KernelAbstractions.adapt(backend, left_bc_vec))

    # flat index: i + (t-1)*nx + (s-1)*nx*nt
    idx(i, t, s) = i + (t-1)*nx + (s-1)*nx*nt

    # --------------------------------------------------
    # 1. Dirichlet BC at left boundary: u(1,t,s) = left_bc[s]
    # --------------------------------------------------
    ts_pairs_all = [(t, s) for t in 1:nt for s in 1:n_samples]
    constraint(core,
        (u_flat[idx(1, d[1], d[2])] - left_bc_param[d[2]] for d in ts_pairs_all);
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



smooth_pos(x, eps) = 0.5 * (x + sqrt(x^2 + eps^2))
smooth_neg(x, eps) = 0.5 * (x - sqrt(x^2 + eps^2))

function burgers_constraints_IC_Mass_Flux!(model::Model, u, u0, nt, n_samples, grid_points, grid_spacing, dt, params=(;))
    nx  = grid_points[1]
    dx = grid_spacing[1]
    k = get(params, :k, 5)
    eps = get(params, :eps, 1e-6)

    λ = dt / dx

    # Register smooth functions for JuMP nonlinear expressions
    register(model, :smooth_pos, 2, smooth_pos; autodiff = true)
    register(model, :smooth_neg, 2, smooth_neg; autodiff = true)

        # Godunov / Engquist–Osher flux for Burgers:
    # F(uL,uR) = 1/2 * max(uL,0)^2 + 1/2 * min(uR,0)^2
    @NLexpression(model, F[i = 1:nx-1, t = 1:nt, s = 1:n_samples],
        0.5 * smooth_pos(u[i, t, s], eps)^2 +
        0.5 * smooth_neg(u[i+1, t, s], eps)^2
    )

    # 1. Initial condition: u(x, 0) = u_IC(x)
    @constraint(model, [i in 1:nx, s in 1:n_samples],
        u[i, 1, s] == u0[i, 1, 1, s]
    )

    # 2. Constant mass: ∫u(x,t)dx = ∫u(x,0)dx
    @constraint(model, [t in 1:nt, s in 1:n_samples],
        sum(u[i, t, s] for i in 1:nx) * dx ==
        sum(u0[i, 1, 1, s] for i in 1:nx) * dx
    )

    # 3. k local Godunov/Euler updates
    k_eff = min(k, nt - 1)

    @NLconstraint(model, [t in 1:k_eff, i in 2:nx-1, s in 1:n_samples],
        u[i, t+1, s] ==
        u[i, t, s] - λ * (F[i, t, s] - F[i-1, t, s])
    )

    return nothing

end


function burgers_constraints_IC_Mass_Flux!(core::ExaCore, u_flat, u0_flat, nt, n_samples, grid_points, grid_spacing, dt, params=(;); backend=CPU())
    nx  = grid_points[1]
    dx  = grid_spacing[1]
    k   = get(params, :k, 5)
    eps = get(params, :eps, 1e-6)
    λ   = dt / dx

    # flat index: i + (t-1)*nx + (s-1)*nx*nt
    idx(i, t, s) = i + (t-1)*nx + (s-1)*nx*nt

    u0_param = parameter(core, u0_flat)

    # --------------------------------------------------
    # 1. Initial condition: u(i, 1, s) == u0(i, s)
    # --------------------------------------------------
    constraint(core,
        (u_flat[idx(i, 1, s)] - u0_param[i, s]
         for i in 1:nx, s in 1:n_samples);
        lcon = KernelAbstractions.adapt(backend, zeros(nx * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros(nx * n_samples))
    )

    # --------------------------------------------------
    # 2. Constant mass: ∑u[i,t,s]*dx == ∑u0[i,s]*dx  for all t, s
    # --------------------------------------------------
    ts_pairs = [(t, s) for t in 1:nt for s in 1:n_samples]
    constraint(core,
        (
            sum(u_flat[idx(i, d[1], d[2])] for i in 1:nx) * dx -
            sum(u0_param[i, d[2]] for i in 1:nx) * dx
            for d in ts_pairs
        );
        lcon = KernelAbstractions.adapt(backend, zeros(nt * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros(nt * n_samples))
    )

    # --------------------------------------------------
    # 3. k local Godunov steps for interior points i in 2:nx-1
    #    u[i,t+1,s] == u[i,t,s] - λ*(F[i,t,s] - F[i-1,t,s])
    #    F[i,t,s] = 0.5*smooth_pos(u[i,t,s], eps)^2 + 0.5*smooth_neg(u[i+1,t,s], eps)^2
    # --------------------------------------------------
    k_eff = min(k, nt - 1)
    tis_pairs = [(t, i, s) for t in 1:k_eff for i in 2:nx-1 for s in 1:n_samples]
    constraint(core,
        (
            u_flat[idx(d[2], d[1]+1, d[3])] - u_flat[idx(d[2], d[1], d[3])] +
            λ * (
                (0.5*smooth_pos(u_flat[idx(d[2],   d[1], d[3])], eps)^2 + 0.5*smooth_neg(u_flat[idx(d[2]+1, d[1], d[3])], eps)^2) -
                (0.5*smooth_pos(u_flat[idx(d[2]-1, d[1], d[3])], eps)^2 + 0.5*smooth_neg(u_flat[idx(d[2],   d[1], d[3])], eps)^2)
            )
            for d in tis_pairs
        );
        lcon = KernelAbstractions.adapt(backend, zeros(k_eff * (nx-2) * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros(k_eff * (nx-2) * n_samples))
    )

    return nothing
end