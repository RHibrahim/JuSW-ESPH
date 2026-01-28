struct GridParams
    ngridx::Int
    ngridy::Int
    hsmlcellx::Int
    hsmlcelly::Int
    cellx_id::Array{Int64,1}
    celly_id::Array{Int64,1}
    next_cell::Array{Int64,1}
    grid_head::Array{Int64,2}
    minx::Float64
    miny::Float64
    geomx::Float64
    geomy::Float64
end

function build_grid(x, y, hsml, skf, ntotal)

    # --------------------------------------------------
    # Grid construction
    # --------------------------------------------------
    cellx_id  = zeros(Int64, ntotal)
    celly_id  = zeros(Int64, ntotal)
    next_cell = zeros(Int64, ntotal)

    scale_k = (skf == 1) ? 2 : 3
    hsmlmax = maximum(hsml)

    max_coordx = maximum(x) + scale_k*hsmlmax
    max_coordy = maximum(y) + scale_k*hsmlmax
    min_coordx = minimum(x) - scale_k*hsmlmax
    min_coordy = minimum(y) - scale_k*hsmlmax

    geomx = max_coordx - min_coordx
    geomy = max_coordy - min_coordy

    nppg = 1    # Averaged number of particles per grid cell
    ngridx = floor(Int64, sqrt(ntotal * geomx / geomy * nppg)) + 1
    ngridy = floor(Int64, ngridx * geomy / geomx) + 1

    hsmlcellx = floor(Int64, ngridx * hsmlmax / geomx) + 1
    hsmlcelly = floor(Int64, ngridy * hsmlmax / geomy) + 1
    grid_head = zeros(Int64, ngridx, ngridy)

    # --------------------------------------------------
    # Insert particles into grid (linked cells)
    # --------------------------------------------------
    for i in 1:ntotal
        cellx_id[i] = floor(Int64, ngridx * (x[i] - min_coordx) / geomx) + 1
        celly_id[i] = floor(Int64, ngridy * (y[i] - min_coordy) / geomy) + 1
        next_cell[i] = grid_head[cellx_id[i], celly_id[i]]
        grid_head[cellx_id[i], celly_id[i]] = i
    end

    return GridParams(ngridx, ngridy, hsmlcellx, hsmlcelly, cellx_id, celly_id, next_cell, grid_head, min_coordx, min_coordy, geomx, geomy)

end