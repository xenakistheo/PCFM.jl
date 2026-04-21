using ExaModels


function heat_constraints!(core, u_flat, params)
    (; Nx, Nt, dx, u0, backend) = params

    # helper indexing (i = space, t = time)
    idx(i, t) = (t-1)*Nx + i

    # --------------------------------------------------
    # 1. Initial condition: u(x,0) = u0(x)
    # --------------------------------------------------
    for i in 1:Nx
        constraint(core, u_flat[idx(i, 1)] == u0[i]) #Is this needed 
    end

    # --------------------------------------------------
    # 2. Mass conservation: ∑ u(x,t) dx = 0 for all t
    # --------------------------------------------------
    for t in 1:Nt
        mass = sum(u_flat[idx(i, t)] for i in 1:Nx) * dx
        constraint(core, mass == 0.0)
    end
    return nothing 
end 

function ns_constraints!(core, u_flat, params)
    (; Nx, Ny, Nt, dx, dy, u0, backend) = params

    # helper index
    idx(i,j,t) = (t-1)*Nx*Ny + (j-1)*Nx + i

    # --------------------------------------------------
    # 1. Initial condition
    # --------------------------------------------------
    for i in 1:Nx, j in 1:Ny
        constraint(core, u_flat[idx(i,j,1)] == u0[i,j])
    end

    # --------------------------------------------------
    # 2. Global mass (vorticity) conservation
    # --------------------------------------------------
    # reference mass at t=1
    M0 = sum(u0[i,j] for i in 1:Nx, j in 1:Ny) * dx * dy

    for t in 2:Nt
        Mt = sum(u_flat[idx(i,j,t)] for i in 1:Nx, j in 1:Ny) * dx * dy
        constraint(core, Mt == M0)
    end

    return nothing
end

function rd_constraints!(core, u_flat, params)
    (; Nx, Nt, dx, dt, rho, nu, u0, backend) = params

    # index helper
    idx(i,t) = (t-1)*Nx + i

    # --------------------------------------------------
    # 1. Initial condition
    # --------------------------------------------------
    for i in 1:Nx
        constraint(core, u_flat[idx(i,1)] == u0[i])
    end

    # --------------------------------------------------
    # Precompute mass and source expressions
    # --------------------------------------------------
    M = Vector{Any}(undef, Nt)
    S = Vector{Any}(undef, Nt)
    F = Vector{Any}(undef, Nt)

    for t in 1:Nt
        # mass
        M[t] = sum(u_flat[idx(i,t)] for i in 1:Nx) * dx

        # reaction term
        S[t] = rho * sum(u_flat[idx(i,t)] * (1 - u_flat[idx(i,t)]) for i in 1:Nx) * dx

        # boundary flux (4th order FD like your code)
        u = i -> u_flat[idx(i,t)]

        gL = -nu * (-25*u(1) + 48*u(2) - 36*u(3) + 16*u(4) - 3*u(5)) / (12*dx)
        gR = -nu * (25*u(Nx) - 48*u(Nx-1) + 36*u(Nx-2) - 16*u(Nx-3) + 3*u(Nx-4)) / (12*dx)

        F[t] = gL - gR
    end

    # --------------------------------------------------
    # 2. Enforce mass evolution constraint
    # --------------------------------------------------
    for t in 2:Nt
        # trapezoidal cumulative integrals
        S_cum = sum(0.5*(S[j] + S[j+1]) * dt for j in 1:t-1)
        F_cum = sum(0.5*(F[j] + F[j+1]) * dt for j in 1:t-1)

        constraint(core, M[t] == M[1] + S_cum + F_cum)
    end

    return nothing
end

function burgers_constraints!(core, u_flat, params)
    (; Nx, Nt, dx, dt, left_bc, backend) = params

    # index helper
    idx(i,t) = (t-1)*Nx + i

    # --------------------------------------------------
    # 1. Boundary conditions
    # --------------------------------------------------
    for t in 1:Nt
        # Dirichlet at left boundary
        constraint(core, u_flat[idx(1,t)] == left_bc)

        # Neumann at right boundary (zero gradient)
        constraint(core, u_flat[idx(Nx,t)] == u_flat[idx(Nx-1,t)])
    end

    # --------------------------------------------------
    # 2. Mass evolution constraint
    # --------------------------------------------------
    # Precompute symbolic expressions
    M = Vector{Any}(undef, Nt)
    F = Vector{Any}(undef, Nt)

    for t in 1:Nt
        # Mass at time t
        M[t] = sum(u_flat[idx(i,t)] for i in 1:Nx) * dx

        # Flux term f(u) = 0.5*u^2
        uL = u_flat[idx(1,t)]
        uR = u_flat[idx(Nx,t)]

        # Matches PyTorch: flux = f[:, -1] - f[:, 0]
        F[t] = 0.5*uR^2 - 0.5*uL^2
    end

    # Enforce integral constraint over time
    for t in 2:Nt
        flux_cum = sum(0.5*(F[j] + F[j+1]) * dt for j in 1:t-1)

        constraint(core, M[t] == M[1] - flux_cum)
    end

    return nothing
end