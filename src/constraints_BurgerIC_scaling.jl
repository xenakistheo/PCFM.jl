using ExaModels
using KernelAbstractions

smooth_pos(x, eps) = 0.5 * (x + sqrt(x^2 + eps^2))
smooth_neg(x, eps) = 0.5 * (x - sqrt(x^2 + eps^2))

function burgers_constraints_IC!(model::Model, u, u0, nt, n_samples, grid_points, grid_spacing, dt, params=(;))
    nx  = grid_points[1]
    dx = grid_spacing[1]
    k = get(params, :k, 5)
    eps = get(params, :eps, 1e-6)

    # 1. Initial condition: u(x, 0) = u_IC(x)
    @constraint(model, [i in 1:nx, s in 1:n_samples],
        u[i, 1, s] == u0[i, 1, 1, s]
    )

    return nothing
end 

function burgers_constraints_IC!(core::ExaCore, u_flat, u0_flat, nt, n_samples, grid_points, grid_spacing, dt, params=(;); backend=CPU())
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

    return nothing
end


function burgers_constraints_IC_Mass(model::Model, u, u0, nt, n_samples, grid_points, grid_spacing, dt, params=(;))
    nx  = grid_points[1]
    dx = grid_spacing[1]
    k = get(params, :k, 5)
    eps = get(params, :eps, 1e-6)

    # 1. Initial condition: u(x, 0) = u_IC(x)
    @constraint(model, [i in 1:nx, s in 1:n_samples],
        u[i, 1, s] == u0[i, 1, 1, s]
    )

    # 2. Constant mass: ∫u(x,t)dx = ∫u(x,0)dx
    @constraint(model, [t in 1:nt, s in 1:n_samples],
        sum(u[i, t, s] for i in 1:nx) * dx ==
        sum(u0[i, 1, 1, s] for i in 1:nx) * dx
    )
    return nothing
end 

function burgers_constraints_IC_Mass!(core::ExaCore, u_flat, u0_flat, nt, n_samples, grid_points, grid_spacing, dt, params=(;); backend=CPU())
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

    return nothing
end


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

