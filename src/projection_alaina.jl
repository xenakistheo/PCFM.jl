#  projection.jl — AbstractProjectionSolver interface for PCFM

using Optimization, OptimizationOptimJL, ForwardDiff
using ADTypes, DifferentiationInterface
abstract type AbstractProjectionSolver end

## N-dimensional helpers
#  4D: (nx, nt, channels, batch)     → 1D spatial
#  5D: (sx, sy, nt, channels, batch) → 2D spatial

function get_array_layout(Z)
    nd = ndims(Z)
    if nd == 4
        return (nt=size(Z,2), nc=size(Z,3), nb=size(Z,4))
    elseif nd == 5
        return (nt=size(Z,3), nc=size(Z,4), nb=size(Z,5))
    else
        error("Unsupported array rank $nd")
    end
end

function get_slice(Z, k, c, b)
    ndims(Z) == 4 ? (@view Z[:, k, c, b]) : (@view Z[:, :, k, c, b])
end

function set_slice!(Z, k, c, b, vals)
    ndims(Z) == 4 ? (Z[:, k, c, b] .= vals) : (Z[:, :, k, c, b] .= vals)
end

function n_spatial(Z)
    ndims(Z) == 4 ? size(Z, 1) : size(Z, 1) * size(Z, 2)
end

# 1D SOLVERS

struct NoOpSolver <: AbstractProjectionSolver end

function solve_projection(::NoOpSolver, x, constraint_data)
    return x
end

struct AnalyticICProjectionSolver <: AbstractProjectionSolver end

function solve_projection(::AnalyticICProjectionSolver, Z_hat, constraint_data)
    Z_proj = copy(Z_hat)
    Z_proj[:, 1:1, :, :] .= constraint_data.u_0_ic_matrix
    return Z_proj
end

struct AnalyticEnergyProjectionSolver <: AbstractProjectionSolver end

function solve_projection(::AnalyticEnergyProjectionSolver, Z_hat, constraint_data)
    nx, nt, nc, nb = size(Z_hat)
    dx = constraint_data.dx
    E0 = sum(abs2, constraint_data.u_0_ic_vec) * dx
    Z_proj = copy(Z_hat)
    for b in 1:nb, c in 1:nc, k in 1:nt
        slice = @view Z_proj[:, k, c, b]
        Ek = sum(abs2, slice) * dx
        if Ek > 0
            slice .*= sqrt(E0 / Ek)
        end
    end
    return Z_proj
end

struct AnalyticICEnergyProjectionSolver <: AbstractProjectionSolver end

function solve_projection(::AnalyticICEnergyProjectionSolver, Z_hat, constraint_data)
    nx, nt, nc, nb = size(Z_hat)
    dx = constraint_data.dx
    E0 = sum(abs2, constraint_data.u_0_ic_vec) * dx
    Z_proj = copy(Z_hat)
    for b in 1:nb, c in 1:nc
        Z_proj[:, 1, c, b] .= constraint_data.u_0_ic_vec
        for k in 2:nt
            slice = @view Z_proj[:, k, c, b]
            Ek = sum(abs2, slice) * dx
            if Ek > 0
                slice .*= sqrt(E0 / Ek)
            end
        end
    end
    return Z_proj
end

struct AnalyticMassProjectionSolver <: AbstractProjectionSolver end

function solve_projection(::AnalyticMassProjectionSolver, Z_hat, constraint_data)
    nx, nt, nc, nb = size(Z_hat)
    dx = constraint_data.dx
    Z_proj = copy(Z_hat)
    for b in 1:nb, c in 1:nc
        Z_proj[:, 1, c, b] .= constraint_data.u_0_ic_vec
        for k in 2:nt
            slice = @view Z_proj[:, k, c, b]
            mass_k = sum(slice) * dx
            slice .-= mass_k / (nx * dx)
        end
    end
    return Z_proj
end

struct AnalyticICMassProjectionSolver <: AbstractProjectionSolver end

function solve_projection(::AnalyticICMassProjectionSolver, Z_hat, constraint_data)
    nx, nt, nc, nb = size(Z_hat)
    dx = constraint_data.dx
    Z_proj = copy(Z_hat)
    for b in 1:nb, c in 1:nc
        Z_proj[:, 1, c, b] .= constraint_data.u_0_ic_vec
        for k in 2:nt
            slice = @view Z_proj[:, k, c, b]
            mass_k = sum(slice) * dx
            slice .-= mass_k / (nx * dx)
        end
    end
    return Z_proj
end

# LBFGS + IP - still 1D

struct PenaltyLBFGSEnergyProjectionSolver{T} <: AbstractProjectionSolver
    penalty::T
end
PenaltyLBFGSEnergyProjectionSolver(; penalty=1.0f4) = PenaltyLBFGSEnergyProjectionSolver(penalty)

function solve_projection(solver::PenaltyLBFGSEnergyProjectionSolver, Z_hat, constraint_data)
    nx, nt, nc, nb = size(Z_hat)
    dx = constraint_data.dx
    E0 = sum(abs2, constraint_data.u_0_ic_vec) * dx
    λ = solver.penalty
    Z_proj = copy(Z_hat)

    function penalised_loss(u_k, p)
        u_hat_k = p[1]
        obj = sum(abs2, u_k .- u_hat_k)
        energy_viol = sum(abs2, u_k) * dx - E0
        return obj + λ * energy_viol^2
    end

    opt_func = OptimizationFunction(penalised_loss, AutoForwardDiff())

    for b in 1:nb, c in 1:nc
        Z_proj[:, 1, c, b] .= constraint_data.u_0_ic_vec
        for k in 2:nt
            u_hat_k = Z_proj[:, k, c, b]
            u0 = copy(u_hat_k)
            prob = OptimizationProblem(opt_func, u0, (u_hat_k,))
            sol = solve(prob, OptimizationOptimJL.LBFGS())
            Z_proj[:, k, c, b] .= sol.u
        end
    end
    return Z_proj
end

struct IPEnergyProjectionSolver <: AbstractProjectionSolver end

function _ip_energy_loss(u_k, p)
    u_hat_k = p[1]
    return sum(abs2, u_k .- u_hat_k)
end

function _ip_energy_cons!(res, u_k, p)
    E0 = p[2]
    dx = p[3]
    res[1] = sum(abs2, u_k) * dx - E0
    return nothing
end

function solve_projection(::IPEnergyProjectionSolver, Z_hat, cdata)
    nx, nt, nc, nb = size(Z_hat)
    dx = cdata.dx
    E0 = sum(abs2, cdata.u_0_ic_vec) * dx
    Z_proj = copy(Z_hat)

    ad_type = DifferentiationInterface.SecondOrder(AutoForwardDiff(), AutoForwardDiff())
    opt_func = OptimizationFunction(_ip_energy_loss, ad_type; cons = _ip_energy_cons!)

    for b in 1:nb, c in 1:nc
        Z_proj[:, 1, c, b] .= cdata.u_0_ic_vec
        for k in 2:nt
            u_hat_k = Float64.(Z_proj[:, k, c, b])
            u0 = copy(u_hat_k)
            p = (u_hat_k, Float64(E0), Float64(dx))
            prob = OptimizationProblem(opt_func, u0, p; lcons = [0.0], ucons = [0.0])
            sol = solve(prob, OptimizationOptimJL.IPNewton())
            Z_proj[:, k, c, b] .= Float32.(sol.u)
        end
    end
    return Z_proj
end

struct PenaltyLBFGSMassProjectionSolver{T} <: AbstractProjectionSolver
    penalty::T
end
PenaltyLBFGSMassProjectionSolver(; penalty=1.0f4) = PenaltyLBFGSMassProjectionSolver(penalty)

function solve_projection(solver::PenaltyLBFGSMassProjectionSolver, Z_hat, constraint_data)
    nx, nt, nc, nb = size(Z_hat)
    dx = constraint_data.dx
    λ = solver.penalty
    Z_proj = copy(Z_hat)

    function penalised_loss(u_k, p)
        u_hat_k = p[1]
        obj = sum(abs2, u_k .- u_hat_k)
        mass_viol = sum(u_k) * dx
        return obj + λ * mass_viol^2
    end

    opt_func = OptimizationFunction(penalised_loss, AutoForwardDiff())

    for b in 1:nb, c in 1:nc
        Z_proj[:, 1, c, b] .= constraint_data.u_0_ic_vec
        for k in 2:nt
            u_hat_k = Z_proj[:, k, c, b]
            u0 = copy(u_hat_k)
            prob = OptimizationProblem(opt_func, u0, (u_hat_k,))
            sol = solve(prob, OptimizationOptimJL.LBFGS())
            Z_proj[:, k, c, b] .= sol.u
        end
    end
    return Z_proj
end

struct IPMassProjectionSolver <: AbstractProjectionSolver end

function _ip_mass_loss(u_k, p)
    u_hat_k = p[1]
    return sum(abs2, u_k .- u_hat_k)
end

function _ip_mass_cons!(res, u_k, p)
    dx = p[2]
    res[1] = sum(u_k) * dx
    return nothing
end

function solve_projection(::IPMassProjectionSolver, Z_hat, cdata)
    nx, nt, nc, nb = size(Z_hat)
    dx = cdata.dx
    Z_proj = copy(Z_hat)

    ad_type = DifferentiationInterface.SecondOrder(AutoForwardDiff(), AutoForwardDiff())
    opt_func = OptimizationFunction(_ip_mass_loss, ad_type; cons = _ip_mass_cons!)

    for b in 1:nb, c in 1:nc
        Z_proj[:, 1, c, b] .= cdata.u_0_ic_vec
        for k in 2:nt
            u_hat_k = Float64.(Z_proj[:, k, c, b])
            u0 = copy(u_hat_k)
            p = (u_hat_k, Float64(dx))
            prob = OptimizationProblem(opt_func, u0, p; lcons = [0.0], ucons = [0.0])
            sol = solve(prob, OptimizationOptimJL.IPNewton())
            Z_proj[:, k, c, b] .= Float32.(sol.u)
        end
    end
    return Z_proj
end

# RD - 1D

struct RDSolver{T} <: AbstractProjectionSolver
    penalty::T
    rho::Float32
end
RDSolver(; penalty=1.0f4, rho=1.0f0) = RDSolver(penalty, rho)

function solve_projection(solver::RDSolver, Z_hat, cdata)
    nx, nt, nc, nb = size(Z_hat)
    dx      = cdata.dx
    dt_phys = cdata.dt_physics
    ρ       = solver.rho
    λ       = solver.penalty
    Z_proj  = copy(Z_hat)

    function penalised_loss(u_next, p)
        u_hat_next, M_target, _dx = p
        obj = sum(abs2, u_next .- u_hat_next)
        M_next = sum(u_next) * _dx
        return obj + λ * (M_next - M_target)^2
    end

    opt_func = OptimizationFunction(penalised_loss, AutoForwardDiff())

    for b in 1:nb, c in 1:nc
        Z_proj[:, 1, c, b] .= cdata.u_0_ic_vec

        for k in 1:(nt - 1)
            u_k = Z_proj[:, k, c, b]
            M_k = sum(u_k) * dx
            source = ρ * sum(u_k[i] * (1.0f0 - u_k[i]) for i in 1:nx) * dx
            flux = 0
            M_target = M_k + dt_phys * (source + flux)

            u_hat_next = Z_proj[:, k+1, c, b]
            u0 = copy(u_hat_next)
            prob = OptimizationProblem(opt_func, u0,
                (u_hat_next, Float32(M_target), dx))
            sol = solve(prob, OptimizationOptimJL.LBFGS())
            Z_proj[:, k+1, c, b] .= sol.u
        end
    end
    return Z_proj
end

struct RDIPNewtonSolver <: AbstractProjectionSolver
    rho::Float32
end
RDIPNewtonSolver(; rho=1.0f0) = RDIPNewtonSolver(rho)

function _rd_ip_loss(u_next, p)
    u_hat_next = p[1]
    return sum(abs2, u_next .- u_hat_next)
end

function _rd_ip_cons!(res, u_next, p)
    M_target = p[2]
    _dx = p[3]
    res[1] = sum(u_next) * _dx - M_target
    return nothing
end

function solve_projection(solver::RDIPNewtonSolver, Z_hat, cdata)
    nx, nt, nc, nb = size(Z_hat)
    dx      = cdata.dx
    dt_phys = cdata.dt_physics
    ρ       = solver.rho
    Z_proj  = copy(Z_hat)

    ad_type = DifferentiationInterface.SecondOrder(AutoForwardDiff(), AutoForwardDiff())
    opt_func = OptimizationFunction(_rd_ip_loss, ad_type; cons = _rd_ip_cons!)

    for b in 1:nb, c in 1:nc
        Z_proj[:, 1, c, b] .= cdata.u_0_ic_vec

        for k in 1:(nt - 1)
            u_k = Z_proj[:, k, c, b]
            M_k = sum(u_k) * dx
            source = ρ * sum(u_k[i] * (1.0f0 - u_k[i]) for i in 1:nx) * dx
            flux = 0
            M_target = M_k + dt_phys * (source + flux)

            u_hat_next = Float64.(Z_proj[:, k+1, c, b])
            u0 = copy(u_hat_next)
            p = (u_hat_next, Float64(M_target), Float64(dx))
            prob = OptimizationProblem(opt_func, u0, p; lcons = [0.0], ucons = [0.0])
            sol = solve(prob, OptimizationOptimJL.IPNewton())
            Z_proj[:, k+1, c, b] .= Float32.(sol.u)
        end
    end
    return Z_proj
end

#  B2 — NS Vorticity: ∫w dx = W₀ - 2D

struct NSVorticityAnalyticSolver <: AbstractProjectionSolver end

function solve_projection(::NSVorticityAnalyticSolver, Z_hat, cdata)
    layout = get_array_layout(Z_hat)
    dx = cdata.dx
    W0 = cdata.M0
    npts = n_spatial(Z_hat)
    Z_proj = copy(Z_hat)
    for b in 1:layout.nb, c in 1:layout.nc
        set_slice!(Z_proj, 1, c, b, cdata.u_0_ic)
        for k in 2:layout.nt
            slice = get_slice(Z_proj, k, c, b)
            W_k = sum(slice) * dx
            slice .-= (W_k - W0) / (npts * dx)
        end
    end
    return Z_proj
end

struct NSVorticityLBFGSSolver{T} <: AbstractProjectionSolver
    penalty::T
end
NSVorticityLBFGSSolver(; penalty=1.0f4) = NSVorticityLBFGSSolver(penalty)

function solve_projection(solver::NSVorticityLBFGSSolver, Z_hat, cdata)
    layout = get_array_layout(Z_hat)
    dx = cdata.dx
    W0 = cdata.M0
    λ  = solver.penalty
    Z_proj = copy(Z_hat)

    function penalised_loss(w_flat, p)
        w_hat_flat, _dx, _W0 = p
        obj = sum(abs2, w_flat .- w_hat_flat)
        W_k = sum(w_flat) * _dx
        return obj + λ * (W_k - _W0)^2
    end

    opt_func = OptimizationFunction(penalised_loss, AutoForwardDiff())

    for b in 1:layout.nb, c in 1:layout.nc
        set_slice!(Z_proj, 1, c, b, cdata.u_0_ic)
        for k in 2:layout.nt
            slice = get_slice(Z_proj, k, c, b)
            w_hat_flat = vec(copy(slice))
            u0 = copy(w_hat_flat)
            prob = OptimizationProblem(opt_func, u0, (w_hat_flat, dx, W0))
            sol = solve(prob, OptimizationOptimJL.LBFGS())
            slice .= reshape(sol.u, size(slice))
        end
    end
    return Z_proj
end

struct NSVorticityIPNewtonSolver <: AbstractProjectionSolver end

function _ns_ip_loss(w_flat, p)
    w_hat_flat = p[1]
    return sum(abs2, w_flat .- w_hat_flat)
end

function _ns_ip_cons!(res, w_flat, p)
    W0 = p[2]
    _dx = p[3]
    res[1] = sum(w_flat) * _dx - W0
    return nothing
end

function solve_projection(::NSVorticityIPNewtonSolver, Z_hat, cdata)
    layout = get_array_layout(Z_hat)
    dx = cdata.dx
    W0 = cdata.M0
    Z_proj = copy(Z_hat)

    ad_type = DifferentiationInterface.SecondOrder(AutoForwardDiff(), AutoForwardDiff())
    opt_func = OptimizationFunction(_ns_ip_loss, ad_type; cons = _ns_ip_cons!)

    for b in 1:layout.nb, c in 1:layout.nc
        set_slice!(Z_proj, 1, c, b, cdata.u_0_ic)
        for k in 2:layout.nt
            slice = get_slice(Z_proj, k, c, b)
            w_hat = Float64.(vec(slice))
            u0 = copy(w_hat)
            p = (w_hat, Float64(W0), Float64(dx))
            prob = OptimizationProblem(opt_func, u0, p; lcons = [0.0], ucons = [0.0])
            sol = solve(prob, OptimizationOptimJL.IPNewton())
            slice .= reshape(Float32.(sol.u), size(slice))
        end
    end
    return Z_proj
end

# Burgers - 1D

# ════════════════════════════════════════════════════════════
#  Burgers Regime 1: BC + Mass (H^{B,BC})
#
#  Constraints per timestep:
#    (a) u(0, t) = u_L                    [left Dirichlet]
#    (b) m(t) = m_0                       [mass conservation]
#    (c) u(N_x, t) - u(N_x-1, t) = 0     [right Neumann via FD]
#
#  Three constraints, nx variables per timestep.
# ════════════════════════════════════════════════════════════

struct BurgersBCMassSolver{T} <: AbstractProjectionSolver
    penalty::T
end
BurgersBCMassSolver(; penalty=1.0f4) = BurgersBCMassSolver(penalty)

function solve_projection(solver::BurgersBCMassSolver, Z_hat, cdata)
    nx, nt, nc, nb = size(Z_hat)
    dx = Float64(cdata.dx)
    λ  = Float64(solver.penalty)
    u_L = Float64(cdata.u_L)
    m0 = Float64(cdata.M0)
    Z_proj = copy(Z_hat)

    function penalised_loss(u_k, p)
        u_hat_k = p[1]
        obj = sum(abs2, u_k .- u_hat_k)
        obj += λ * (u_k[1] - u_L)^2
        mass_viol = sum(u_k) * dx - m0
        obj += λ * mass_viol^2
        neumann_viol = u_k[end] - u_k[end-1]
        obj += λ * neumann_viol^2
        return obj
    end

    opt_func = OptimizationFunction(penalised_loss, AutoForwardDiff())

    for b in 1:nb, c in 1:nc
        for k in 1:nt
            u_hat_k = Float64.(Z_proj[:, k, c, b])
            u0 = copy(u_hat_k)
            prob = OptimizationProblem(opt_func, u0, (u_hat_k,))
            sol = solve(prob, OptimizationOptimJL.LBFGS())
            Z_proj[:, k, c, b] .= Float32.(sol.u)
        end
    end
    return Z_proj
end

# IPNewton version

struct BurgersBCMassIPSolver <: AbstractProjectionSolver end

function _burgers_bc_loss(u_k, p)
    u_hat_k = p[1]
    return sum(abs2, u_k .- u_hat_k)
end

function _burgers_bc_cons!(res, u_k, p)
    u_L = p[2]
    m0 = p[3]
    _dx = p[4]
    # (a) u(0,t) = u_L
    res[1] = u_k[1] - u_L
    # (b) ∫u dx = m₀
    res[2] = sum(u_k) * _dx - m0
    # (c) u(Nx) - u(Nx-1) = 0
    res[3] = u_k[end] - u_k[end-1]
    return nothing
end

function solve_projection(::BurgersBCMassIPSolver, Z_hat, cdata)
    nx, nt, nc, nb = size(Z_hat)
    dx = cdata.dx
    u_L = cdata.u_L
    m0 = cdata.M0
    Z_proj = copy(Z_hat)

    ad_type = DifferentiationInterface.SecondOrder(AutoForwardDiff(), AutoForwardDiff())
    opt_func = OptimizationFunction(_burgers_bc_loss, ad_type; cons = _burgers_bc_cons!)

    for b in 1:nb, c in 1:nc
        for k in 1:nt
            u_hat_k = Float64.(Z_proj[:, k, c, b])
            u0 = copy(u_hat_k)
            p = (u_hat_k, Float64(u_L), Float64(m0), Float64(dx))
            prob = OptimizationProblem(opt_func, u0, p;
                lcons = [0.0, 0.0, 0.0], ucons = [0.0, 0.0, 0.0])
            sol = solve(prob, OptimizationOptimJL.IPNewton())
            Z_proj[:, k, c, b] .= Float32.(sol.u)
        end
    end
    return Z_proj
end

# ════════════════════════════════════════════════════════════
#  Burgers Regime 2: IC + Mass + Local Flux (H^{B,IC})
#
#  Constraints:
#    (a) u(x,0) = u_IC(x; p_loc)          [initial condition]
#    (b) m(t) = m_0                        [mass conservation]
#    (c) R^(k)_Flux(u) = 0                 [Godunov flux residual]
#
#  The Godunov flux residual at interior cells:
#    u_{k+1}[i] = u_k[i] - (dt/dx)(F_{i+1/2} - F_{i-1/2})
#
#  where F is the Godunov numerical flux:
#    F(uL, uR) = max(f(max(uL,0)), f(min(uR,0))) if uL >= uR (shock)
#              = min(f(uL), f(uR))                 if uL < uR (rarefaction)
#  with f(u) = u²/2
#
#  This couples timesteps k and k+1 sequentially.
#  We freeze u_k, enforce the flux residual on u_{k+1}.
# ════════════════════════════════════════════════════════════

struct BurgersICFluxSolver{T} <: AbstractProjectionSolver
    penalty::T
end
BurgersICFluxSolver(; penalty=1.0f4) = BurgersICFluxSolver(penalty)

# Godunov flux for f(u) = u²/2
function godunov_flux(uL, uR)
    if uL >= uR
        # Shock: Rankine-Hugoniot speed s = (uL+uR)/2
        s = (uL + uR) / 2
        return s > 0 ? 0.5f0 * uL^2 : 0.5f0 * uR^2
    else
        # Rarefaction: min of fluxes (sonic point at u=0)
        if uL >= 0
            return 0.5f0 * uL^2
        elseif uR <= 0
            return 0.5f0 * uR^2
        else
            return 0.0f0  # sonic rarefaction
        end
    end
end

function solve_projection(solver::BurgersICFluxSolver, Z_hat, cdata)
    nx, nt, nc, nb = size(Z_hat)
    dx = Float64(cdata.dx)
    dt_phys = Float64(cdata.dt_physics)
    m0 = Float64(cdata.M0)
    λ  = Float64(solver.penalty)
    Z_proj = copy(Z_hat)

    for b in 1:nb, c in 1:nc
        Z_proj[:, 1, c, b] .= cdata.u_0_ic_vec

        for k in 1:(nt - 1)
            u_k = Float64.(Z_proj[:, k, c, b])

            F = zeros(Float64, nx - 1)
            for i in 1:(nx - 1)
                F[i] = godunov_flux(u_k[i], u_k[i+1])
            end

            u_target = copy(u_k)
            cfl = dt_phys / dx
            for i in 2:(nx - 1)
                u_target[i] = u_k[i] - cfl * (F[i] - F[i-1])
            end
            u_target[end] = u_target[end-1]

            function penalised_loss(u_next, p)
                u_hat_next, _u_target, _dx, _m0 = p
                obj = sum(abs2, u_next .- u_hat_next)
                for i in 2:(length(u_next) - 1)
                    obj += λ * (u_next[i] - _u_target[i])^2
                end
                mass_viol = sum(u_next) * _dx - _m0
                obj += λ * mass_viol^2
                return obj
            end

            opt_func = OptimizationFunction(penalised_loss, AutoForwardDiff())

            u_hat_next = Float64.(Z_proj[:, k+1, c, b])
            u0 = copy(u_hat_next)
            prob = OptimizationProblem(opt_func, u0,
                (u_hat_next, u_target, dx, m0))
            sol = solve(prob, OptimizationOptimJL.LBFGS())
            Z_proj[:, k+1, c, b] .= Float32.(sol.u)
        end
    end
    return Z_proj
end

# IPNewton version

struct BurgersICFluxIPSolver <: AbstractProjectionSolver end

function solve_projection(::BurgersICFluxIPSolver, Z_hat, cdata)
    nx, nt, nc, nb = size(Z_hat)
    dx = cdata.dx
    dt_phys = cdata.dt_physics
    m0 = cdata.M0
    Z_proj = copy(Z_hat)

    for b in 1:nb, c in 1:nc
        # (a) Fix IC
        Z_proj[:, 1, c, b] .= cdata.u_0_ic_vec

        for k in 1:(nt - 1)
            u_k = Z_proj[:, k, c, b]

            # Compute Godunov fluxes
            F = zeros(Float64, nx - 1)
            for i in 1:(nx - 1)
                F[i] = godunov_flux(u_k[i], u_k[i+1])
            end

            # Target interior values from Godunov
            cfl = dt_phys / dx
            u_target_interior = zeros(Float64, nx - 2)
            for i in 2:(nx - 1)
                u_target_interior[i-1] = u_k[i] - cfl * (F[i] - F[i-1])
            end

            # n_interior = nx-2 flux constraints + 1 mass constraint = nx-1 total
            n_cons = (nx - 2) + 1

            function _flux_loss(u_next, p)
                u_hat_next = p[1]
                return sum(abs2, u_next .- u_hat_next)
            end

            function _flux_cons!(res, u_next, p)
                _u_target_int = p[2]
                _dx = p[3]
                _m0 = p[4]
                # Interior flux residuals: u_{k+1}[i] - target[i] = 0
                for i in 1:(length(_u_target_int))
                    res[i] = u_next[i+1] - _u_target_int[i]
                end
                # Mass: ∫u dx - m₀ = 0
                res[length(_u_target_int) + 1] = sum(u_next) * _dx - _m0
                return nothing
            end

            ad_type = DifferentiationInterface.SecondOrder(AutoForwardDiff(), AutoForwardDiff())
            opt_func = OptimizationFunction(_flux_loss, ad_type; cons = _flux_cons!)

            u_hat_next = Float64.(Z_proj[:, k+1, c, b])
            u0 = copy(u_hat_next)
            p = (u_hat_next, u_target_interior, Float64(dx), Float64(m0))
            prob = OptimizationProblem(opt_func, u0, p;
                lcons = zeros(n_cons), ucons = zeros(n_cons))
            sol = solve(prob, OptimizationOptimJL.IPNewton())
            Z_proj[:, k+1, c, b] .= Float32.(sol.u)
        end
    end
    return Z_proj
end

# struct BurgersConservationSolver{T} <: AbstractProjectionSolver
#     penalty::T
# end
# BurgersConservationSolver(; penalty=1.0f4) = BurgersConservationSolver(penalty)

# function solve_projection(solver::BurgersConservationSolver, Z_hat, cdata)
#     nx, nt, nc, nb = size(Z_hat)
#     dx      = cdata.dx
#     dt_phys = cdata.dt_physics
#     λ       = solver.penalty
#     Z_proj  = copy(Z_hat)

#     function penalised_loss(u_k, p)
#         u_hat_k, M_target, _dx = p
#         obj = sum(abs2, u_k .- u_hat_k)
#         flux_imbalance = u_k[end]^2 - u_k[1]^2
#         obj += λ * flux_imbalance^2
#         M_k = sum(u_k) * _dx
#         obj += λ * (M_k - M_target)^2
#         return obj
#     end

#     opt_func = OptimizationFunction(penalised_loss, AutoForwardDiff())

#     for b in 1:nb, c in 1:nc
#         Z_proj[:, 1, c, b] .= cdata.u_0_ic_vec

#         for k in 1:(nt - 1)
#             u_prev = Z_proj[:, k, c, b]
#             M_prev = sum(u_prev) * dx
#             flux = -0.5f0 * (u_prev[end]^2 - u_prev[1]^2)
#             M_target = M_prev + dt_phys * flux

#             u_hat_next = Z_proj[:, k+1, c, b]
#             u0 = copy(u_hat_next)
#             prob = OptimizationProblem(opt_func, u0,
#                 (u_hat_next, Float32(M_target), dx))
#             sol = solve(prob, OptimizationOptimJL.LBFGS())
#             Z_proj[:, k+1, c, b] .= sol.u
#         end
#     end
#     return Z_proj
# end

# struct BurgersIPNewtonSolver <: AbstractProjectionSolver end

# function _burgers_ip_loss(u_k, p)
#     u_hat_k = p[1]
#     return sum(abs2, u_k .- u_hat_k)
# end

# function _burgers_ip_cons!(res, u_k, p)
#     M_target = p[2]
#     _dx = p[3]
#     res[1] = u_k[end]^2 - u_k[1]^2
#     res[2] = sum(u_k) * _dx - M_target
#     return nothing
# end

# function solve_projection(::BurgersIPNewtonSolver, Z_hat, cdata)
#     nx, nt, nc, nb = size(Z_hat)
#     dx      = cdata.dx
#     dt_phys = cdata.dt_physics
#     Z_proj  = copy(Z_hat)

#     ad_type = DifferentiationInterface.SecondOrder(AutoForwardDiff(), AutoForwardDiff())
#     opt_func = OptimizationFunction(_burgers_ip_loss, ad_type; cons = _burgers_ip_cons!)

#     for b in 1:nb, c in 1:nc
#         Z_proj[:, 1, c, b] .= cdata.u_0_ic_vec

#         for k in 1:(nt - 1)
#             u_prev = Z_proj[:, k, c, b]
#             M_prev = sum(u_prev) * dx
#             flux = -0.5f0 * (u_prev[end]^2 - u_prev[1]^2)
#             M_target = M_prev + dt_phys * flux

#             u_hat_next = Float64.(Z_proj[:, k+1, c, b])
#             u0 = copy(u_hat_next)
#             p = (u_hat_next, Float64(M_target), Float64(dx))
#             prob = OptimizationProblem(opt_func, u0, p; lcons = [0.0, 0.0], ucons = [0.0, 0.0])
#             sol = solve(prob, OptimizationOptimJL.IPNewton())
#             Z_proj[:, k+1, c, b] .= Float32.(sol.u)
#         end
#     end
#     return Z_proj
# end

#  Constraint data factory
#
#  Returns BOTH old field names (u_0_ic_vec, u_0_ic_matrix)
#  AND new field name (u_0_ic) so all solvers work.

function make_constraint_data(u_0_ic_input, nx_or_spatial, nt, n_samples;
                               dx = nothing,
                               dt_physics = Float32(1.0 / (nt - 1)),
                               g_L = nothing,
                               g_R = nothing,
                               u_L = nothing)
    ic = Float32.(u_0_ic_input)

    if dx === nothing
        nx_val = nx_or_spatial isa Tuple ? prod(nx_or_spatial) : nx_or_spatial
        dx = Float32(2π / nx_val)
    end

    M0 = sum(ic) * dx

    _g_L = g_L === nothing ? zeros(Float32, nt) : g_L
    _g_R = g_R === nothing ? zeros(Float32, nt) : g_R
    _u_L = u_L === nothing ? ic[1] : Float32(u_L)  # default: IC value at left boundary

    # For 1D need to build the old matrix format 
    if ndims(ic) == 1
        nx = length(ic)
        ic_matrix = repeat(reshape(ic, nx, 1, 1, 1), 1, 1, 1, n_samples)
    else
        # 2D IC — matrix format not used by NS solvers - placeholder
        ic_matrix = nothing
    end

    return (
        # New field names (used by NS solvers)
        u_0_ic       = ic,
        # Old field names (used by all 1D solvers)
        u_0_ic_vec   = ndims(ic) == 1 ? ic : vec(ic),
        u_0_ic_matrix = ic_matrix,
        # Shared fields
        dx           = dx,
        dt_physics   = dt_physics,
        nt           = nt,
        M0           = M0,
        g_L          = _g_L,
        g_R          = _g_R,
        u_L          = _u_L,
    )
end