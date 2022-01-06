using DelimitedFiles, Plots, Interpolations

"Heuristic number of threads per block in x-dim."
const tx = 32

"Heuristic number of threads per block in y-dim."
const ty = 8

"Input parameters structure"
mutable struct InputParams
    ybed
    ysurf
    xc
    yc
    xv
    yv
    lx_ly
    nx::Int
    ny::Int
    α
end


"""
    read_data(dat_file::String; resol::Int=128, olen::Int=1, visu=false)

Read bedrock and ice surface elevation data from a text file and returns the interpolated data.

# Arguments
- `dat_file::String`: input data file with 3 space-limited columns as `[x-coord  z-bedrock  z-surface]`
- `resol::Int=128`: output resolution
- `olen::Int=1`: overlength for arrays larger then `nx` and `ny`
"""
function read_data(dat_file::String; resol::Int=128, olen::Int=1)
    
    println("Loading the data ... ")
    data = readdlm(dat_file, Float64)

    xv_d   = data[:,1]
    bed_d  = data[:,2]
    surf_d = data[:,3]

    resol  = resol > tx ? resol : tx
    shiftx = resol % tx
    resx   = (shiftx < tx/2 ? Int(resol - shiftx) : Int(resol + tx - shiftx)) - olen

    xv     = LinRange(xv_d[1], xv_d[end], resol)
    itp1   = interpolate( (xv_d,), bed_d[:,1] , Gridded(Linear()))
    itp2   = interpolate( (xv_d,), surf_d[:,1], Gridded(Linear()))
    bed    = itp1.(xv)
    surf   = itp2.(xv)
    
    @assert length(xv) == size(bed)[1] == size(surf)[1]
    println("Interpolating original data (nxv=$(size(bed_d)[1])) on nxv=$(size(bed)[1]) grid.")
    println("done.")
    return xv, bed, surf
end


"""
    rotate(ybed, ysurf, lin_fit)

Rotate bedrock `ybed` and surface `ysurf` profiles. 
"""
function rotate(ybed, ysurf, lin_fit)
    ybedr  = ybed  .- lin_fit
    ysurfr = ysurf .- lin_fit
    ymin   = minimum(ybedr)
    ybedr  = ybedr  .- ymin
    ysurfr = ysurfr .- ymin
    return ybedr, ysurfr
end


"""
    ls_fit(xv, ybed, ysurf)

Perform linear least-square fit of mean bedrock and surface data.
"""
function ls_fit(xv, ybed, ysurf)
    nxv     = length(xv)
    x_mean  = sum(xv) ./ nxv
    y_avg   = (ybed .+ ysurf) ./ 2.0
    y_mean  = sum(y_avg) ./ nxv
    α       = sum((xv .- x_mean) .* (y_avg .- y_mean)) ./ sum((xv .- x_mean).^2)
    orig    = y_mean .- α .* x_mean
    lin_fit = α.*xv .+ orig
    return lin_fit, y_avg, orig, α
end


"""
    preprocess(xv, ybed, ysurf; do_rotate=false, fact_ny::Int=4, olen::Int=1, visu=false)

Preprocess input data for iceflow model.

# Arguments
- `xv`: x-dim vertice coordinates
- `ybed`: bedrock elevation
- `ysurf`: surface elevation
- `do_rotate=false`: rotate domain to minimise non-used grid-points
- `fact_ny::Int=4`: grid-point increase in y-dim
- `olen::Int=1`: overlength for arrays larger then `nx` and `ny`
"""
function preprocess(xv, ybed, ysurf; do_rotate=false, fact_ny::Int=4, olen::Int=1)
    
    println("Starting preprocessing ... ")
    lin_fit, y_avg, orig, α = ls_fit(xv, ybed, ysurf)

    if do_rotate
        ybedr, ysurfr = rotate(ybed, ysurf, lin_fit)
    else
        ybedr, ysurfr = ybed, ysurf
    end

    ymin   = minimum(ybedr)
    ybedr  = ybedr  .- ymin
    ysurfr = ysurfr .- ymin
    ly     = maximum(ysurfr)
    lx     = maximum(xv) - minimum(xv)
    xc     = 0.5(xv[1:end-1].+xv[2:end])
    nx     = length(xc)
    resy   = ceil(Int, ly/lx*nx)
    resy   = resy > tx ? resy : tx
    shifty = resy % ty
    ny     = fact_ny * (shifty < ty/2 ? Int(resy - shifty) : Int(resy + ty - shifty)) - olen
    nyv    = ny+1
    yv     = LinRange(0,ly,nyv)
    yc     = 0.5(yv[1:end-1].+yv[2:end])

    println("Preprocessed data: nx=$nx, ny=$ny (dx=$(lx/nx), dx=$(ly/ny))")

    inputs = InputParams(ybedr/ly, ysurfr/ly, xc/ly, yc/ly, xv/ly, yv/ly, lx/ly, nx, ny, α) # nondim
    # inputs = InputParams(ybedr, ysurfr, xc, yc, xv, yv, lx, nx, ny, α) # dim
    
    println("done.")
    return inputs, lin_fit, y_avg, orig
end

