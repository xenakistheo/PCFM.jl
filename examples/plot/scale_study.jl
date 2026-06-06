using CairoMakie

labels = ["EXA/MadNLP (GPU)", "EXA-CPU", "JuMP/MadNLP", "JuMP/IPOPT"]
colors = [:blue, :red, :green, :orange]
markers = [:circle, :rect, :diamond, :utriangle]

nsample_vals = [8, 16, 32, 64, 128, 256]

times = [
    [3.66, 5.93, 11.58, 21.97, 43.69, 100.27],
    [3.20, 6.80, 14.34, 28.26, 55.93, 120.98],
    [28.27, 59.06, 138.54, 307.11, 598.57, 1350.31],
    [40.29, 79.20, 184.68, 391.74, 794.89, 1734.87],
]

stds = [
    [0.09, 0.11, 0.06, 0.42, 0.92, 1.24],
    [0.03, 0.45, 0.09, 1.39, 1.25, 5.39],
    [0.29, 3.40, 4.86, 11.78, 8.52, 179.46],
    [0.89, 3.80, 2.07, 14.20, 5.86, 188.17]
]

fig = Figure(size = (850, 550))

ax = Axis(
    fig[1, 1],
    xlabel = "Number of samples",
    ylabel = "Runtime [s]",
    title = "Runtime scaling with number of samples",
    xscale = log2,
    yscale = log10,
    xticks = nsample_vals,
    xlabelsize = 20,
    ylabelsize = 20,
    titlesize = 22,
    xticklabelsize = 16,
    yticklabelsize = 16,
)

for i in eachindex(labels)
    lines!(
        ax,
        nsample_vals,
        times[i],
        color = colors[i],
        linewidth = 2.5,
        label = labels[i],
    )

    scatter!(
        ax,
        nsample_vals,
        times[i],
        color = colors[i],
        marker = markers[i],
        markersize = 10,
    )

    errorbars!(
        ax,
        nsample_vals,
        times[i],
        stds[i],
        color = colors[i],
        whiskerwidth = 10,
        linewidth = 1.5,
    )
end

axislegend(ax, position = :lt, labelsize = 16)

fig

save("examples/plot/finalPlots/nsample_scale.png", fig)