function visualize(x, y,int_fluid)
    
    # ------------------------------------------------------------------
    # Use ONLY fluid particles (exclude virtual & open boundaries)
    # ------------------------------------------------------------------
    xx = x[int_fluid]
    yy = y[int_fluid]
    
    #num = readdlm("benchmarks/$(case)/Data_at_time_$(round(final_time,digits=2)).dat")
    base_files = glob("benchmarks/$(case)/Data_at_time_[0-9]*.dat")

    # exclude *_centerline.dat
    filter!(f -> !endswith(f, "_centerline.dat"), base_files)

    @assert !isempty(base_files) "No base output files found"

    sort!(base_files)
    base_fname = last(base_files)

    println("Reading base file: $base_fname")
    
    fname_only = basename(base_fname)

    m = match(r"Data_at_time_([0-9]+(?:\.[0-9]+)?)\.dat", fname_only)
    @assert m !== nothing "Could not extract time from $fname_only"

    final_time = parse(Float64, m.captures[1])

    println("Final simulation time = ", final_time)

    num = readdlm(base_fname)
    wse = num[:, 6]
    h   = num[:, 7]
    hh  = wse[:]   # or h[int_fluid] if you want water depth

    # ------------------------------------------------------------------
    # Build grid from fluid particles only
    # ------------------------------------------------------------------
    dx = minimum(diff(sort(unique(xx))))
    dy = minimum(diff(sort(unique(yy))))

    xg = collect(range(minimum(xx), maximum(xx), step=dx))
    yg = collect(range(minimum(yy), maximum(yy), step=dy))

    nx = length(xg)
    ny = length(yg)

    # ------------------------------------------------------------------
    # Box-averaged field
    # ------------------------------------------------------------------
    H = fill(NaN, ny, nx)
    count = zeros(Int, ny, nx)

    for k in eachindex(hh)
        ix = Int(floor((xx[k] - xg[1]) / dx)) + 1
        iy = Int(floor((yy[k] - yg[1]) / dy)) + 1

        if 1 ≤ ix ≤ nx && 1 ≤ iy ≤ ny
            H[iy, ix] = isnan(H[iy, ix]) ? hh[k] : H[iy, ix] + hh[k]
            count[iy, ix] += 1
        end
    end

    # Average if multiple fluid particles fall in the same box
    for i in 1:nx, j in 1:ny
        if count[j, i] > 0
            H[j, i] /= count[j, i]
        end
    end

    # Optional: mask dry cells
    H[H .< 1e-8] .= NaN

    cs = cgrad(ColorSchemes.davos.colors; rev = true)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    plt = heatmap(
        xg, yg, H;
        c = cs,
        colorbar = true,
        clims = (0.0, maximum(H)),
        xlabel = "Lx",
        ylabel = "Ly",
        title = "$(case) (time = $(round(final_time, digits=2)))",
        framestyle = :box,
        xlims = (minimum(xg), maximum(xg)),
        ylims = (minimum(yg), maximum(yg)),
        size  = (1600, 700),
        aspect_ratio = :auto
    )

    savefig(
        plt,
        "benchmarks/$(case)/$(case)_t$(round(final_time, digits=2)).png"
    )
    
    # Save figure in centerline
    centerline_fname = replace(base_fname, ".dat" => "_centerline.dat")

    @assert isfile(centerline_fname) "Centerline file not found"

    println("Reading centerline file: $centerline_fname")

    num = readdlm(centerline_fname)

    #num = readdlm(centerline_file)

    x_num = num[:, 2]
    h_num = num[:, 6]
    z_b   = num[:, 8]

    p = sortperm(x_num)
    x_num = x_num[p]
    h_num = h_num[p]
    z_b   = z_b[p]

    anafile = "benchmarks/$(case)/analytical_solution.dat"

    ana = readdlm(anafile)

    x_ana = ana[:, 1]
    h_ana = ana[:, 2]

    plt = plot(
        x_ana, h_ana;
        label = "Analytical",
        linewidth = 2,
        linestyle = :solid,
        color = :black
    )

    plt2 = plot!(
        plt,
        x_num, z_b;
        label = "Bottom",
        linewidth = 2,
        linestyle = :solid,
        color = :red
    )

    plot!(
        plt2,
        x_num, h_num;
        label = "JuSW-ESPH solution",
        linewidth = 2,
        linestyle = :solid,
        color = :blue
        #seriestype = :scatter,
        #markersize = 4,
        #markerstrokewidth = 0
    )

    xlabel!(plt, "x")
    ylabel!(plt, "Water surface elevation H")
    title!(plt, "$(case) – at center (t = $(round(final_time, digits=2)))")

    plot!(
        plt,
        framestyle = :box,
        legend = :topright
    )

    savefig(
        plt,
        "benchmarks/$(case)/centerline_H_t$(round(final_time, digits=2)).png"
    )
    

    return nothing
    
end