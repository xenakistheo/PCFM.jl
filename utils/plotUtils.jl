using CairoMakie
CairoMakie.activate!()


function plot_evolution(u_ref, u_proj, Nt, title_text="")
    f = Figure(size = (3000, 1500))

    # Define each Axis explicitly in the grid [row, col]
    ax1 = Axis(f[1, 1], title = "t = 1")
    ax2 = Axis(f[1, 2], title = "t = $(Nt÷4)")
    ax3 = Axis(f[1, 3], title = "t = $(2*Nt÷4)")
    ax4 = Axis(f[1, 4], title = "t = $(3*Nt÷4)")
    ax5 = Axis(f[1, 5], title = "t = end")

    # Plotting on ax1
    lines!(ax1, X, u_proj[:, 1], color = :blue)
    lines!(ax1, X, u_ref[:, 1], color = :red, linestyle = :dash)

    # Plotting on ax2
    lines!(ax2, X, u_proj[:, Nt÷4], color = :blue)
    lines!(ax2, X, u_ref[:, Nt÷4], color = :red, linestyle = :dash)

    # Plotting on ax3
    lines!(ax3, X, u_proj[:, 2*Nt÷4], color = :blue)
    lines!(ax3, X, u_ref[:, 2*Nt÷4], color = :red, linestyle = :dash)

    # Plotting on ax4
    lines!(ax4, X, u_proj[:, 3*Nt÷4], color = :blue)
    lines!(ax4, X, u_ref[:, 3*Nt÷4], color = :red, linestyle = :dash)

    # Plotting on ax5
    lines!(ax5, X, u_proj[:, end], color = :blue)
    lines!(ax5, X, u_ref[:, end], color = :red, linestyle = :dash)

    # Add a Super-Title
    Label(f[0, :], title_text)

    return f
end


function plot_heatmap_evolution(X, T, u_ref, u_sol; title_text="Evolution Heatmap")
    f = Figure(size = (1200, 600))

    ax1 = Axis(f[1, 1], 
              title = "Reference",
              xlabel = "Time", 
              ylabel = "X")
    
    ax2 = Axis(f[1, 2], 
              title = "Solution",
              xlabel = "Time", 
              ylabel = "X")



    heatmap!(ax1, T, X, u_ref', colormap = :viridis)
    heatmap!(ax2, T, X, u_sol', colormap = :viridis)
    Colorbar(f[1, 3], label = "Amplitude")
    Label(f[0, :], title_text)

    return f
end

function constraint_deviation_plot(u_ref, u_proj, T, H, params=nothing; title="")
    f = Figure()
    ax = Axis(
        f[1,1],
        title=title, 
        xlabel = "Time",
    )
    h_ref, h_sol = H(u_ref), H(u_proj)
    lines!(ax, T[1:length(h_ref)], h_ref, label="Reference", )
    lines!(ax, T[1:length(h_sol)], h_sol, label="Solution", linestyle=:dash)
    axislegend(position = :rb)
    
    return f
end

function constraint_deviation_plot(u, time, H, params=nothing; title="")
    f = Figure()
    ax = Axis(
        f[1,1],
        title=title, 
        xlabel = "Time",
    )
    h_ref = H(u)
    lines!(ax, T[1:length(h)], h_sol, label="Solution", linestyle=:dash)
    axislegend(position = :rb)
    return f
end


function plot_sample(k, u1, u2, u3, title="")
    f = Figure(size = (1800, 600))

    ax1 = Axis(f[1, 1], 
                title = "New PCFM",
                xlabel = "Time", 
                ylabel = "X")

    ax2 = Axis(f[1, 2], 
                title = "Old PCFM",
                xlabel = "Time", 
                ylabel = "X")

    ax3 = Axis(f[1, 3], 
                title = "FFM",
                xlabel = "Time", 
                ylabel = "X")


    heatmap!(ax1, T, X, u1[:,:,1,k]', colormap = :viridis)
    heatmap!(ax2, T, X, u2[:,:,1,k]', colormap = :viridis)
    heatmap!(ax3, T, X, u3[:,:,1,k]', colormap = :viridis)


    Colorbar(f[1, 4], label = "Amplitude")
    Label(f[0, :], title)

    return f
end 

function plot_sample(frame, solutions, titles; suptitle=nothing)
    f = Figure(size = (2400, 600))
    N = size(solutions)[1]
    @assert N == length(titles)

    axes = []
    for i in 1:N
        ax = Axis(f[1, i], 
                title = titles[i],
                xlabel = "Time", 
                ylabel = "X")
        push!(axes, ax)
    end 

    for i in 1:N
        heatmap!(axes[i], T, X, solutions[i][:,:,1,frame]', colormap = :viridis)
    end 


    if !isnothing(suptitle)
        Label(f[0, :], suptitle, fontsize=20, font=:bold)
    end
    
    return f
end 


function plot_constraint_violation(frame, solutions, H, titles; constraint_params=nothing, suptitle=nothing)
    f = Figure(size = (2400, 600))
    N = size(solutions)[1]
    @assert N == length(titles)

    axes = []
    for i in 1:N
        ax = Axis(f[1, i], 
                title = titles[i],
                xlabel = "Time", 
                ylabel = "X")
        push!(axes, ax)
    end 

    for i in 1:N
        lines!(axes[i], H(solutions[i][:,:,1,frame], constraint_params))
    end

    if !isnothing(suptitle)
        Label(f[0, :], suptitle, fontsize=20, font=:bold)
    end
    return f
end


function plot_constraint_violation(k, u1, u2, u3, H; title="", constraint_params=nothing)
    f = Figure(size = (1800, 600))

    ax1 = Axis(f[1, 1], 
                title = "New PCFM",
                xlabel = "Time", 
                ylabel = "Violation")

    ax2 = Axis(f[1, 2], 
                title = "Old PCFM",
                xlabel = "Violation", 
                ylabel = "X")

    ax3 = Axis(f[1, 3], 
                title = "FFM",
                xlabel = "Time", 
                ylabel = "Violation")


    lines!(ax1, H(u1[:,:,1,k], constraint_params))
    lines!(ax2, H(u2[:,:,1,k], constraint_params))
    lines!(ax3, H(u3[:,:,1,k], constraint_params))


    Label(f[0, :], title)

    return f
end 
