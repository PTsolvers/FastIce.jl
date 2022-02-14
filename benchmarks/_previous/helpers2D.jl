using DelimitedFiles, Plots, Interpolations

"Heuristic number of threads per block in x-dim."
const tx = 32

"Heuristic number of threads per block in y-dim."
const ty = 8

"Input parameters structure"
mutable struct InputParams2D
    ybedv
    ybedc
    ysurfv
    ysurfc
    xv
    yv
    xc
    yc
    lx_ly
    max∆y
    nx::Int
    ny::Int
    α
end


"Average"
av(A) = 0.5(A[1:end-1].+A[2:end])


"""
    gpu_res(resol, t, olen)

Round the number of grid points optimally for GPUs.
"""
function gpu_res(resol, t, olen)
    resol = resol > t ? resol : t
    shift = resol % t
    return (shift < t/2 ? Int(resol - shift) : Int(resol + t - shift)) - olen
end


"""
    interp(A, B, xv_d, xv)

Linear interpolation of fields `A`, `B`, and `C` on `(xv, yv)` grid.
"""
@views function interp(A, B, xv_d, xv)

    itp1 = interpolate( (xv_d,), A, Gridded(Linear()))
    itp2 = interpolate( (xv_d,), B, Gridded(Linear()))
    
    A2   = itp1.(xv)
    B2   = itp2.(xv)
    
    @assert length(xv) == size(A2)[1] == size(B2)[1]
    return A2, B2
end


"""
    lsq_fit(y_avg, xv)

Linear least-square fit of mean bedrock and surface data.
"""
function lsq_fit(y_avg, xv)
    nxv     = length(xv)
    x_mean  = sum(xv) ./ nxv
    y_mean  = sum(y_avg) ./ nxv
    α       = sum((xv .- x_mean) .* (y_avg .- y_mean)) ./ sum((xv .- x_mean).^2)
    orig    = y_mean .- α .* x_mean
    lin_fit = α.*xv .+ orig
    return lin_fit, α
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
    read_data(dat_file::String, do_interp::Bool, resol::Int, olen::Int)

Read bedrock and ice surface elevation data from a text file and returns the interpolated data.

# Arguments
- `dat_file::String`: input data file with 3 space-limited columns as `[x-coord  z-bedrock  z-surface]`
- `do_interp::Bool`: perform linear interpolation of data
- `resol::Int`: output resolution
- `olen::Int`: overlength for arrays larger then `nx` and `ny`
"""
@views function load_data(dat_file::String, do_interp::Bool, resol::Int, olen::Int)
    
    println("- Loading the data")
    data = readdlm(dat_file, Float64)

    xv_d   = data[:,1]
    bed_d  = data[:,2]
    surf_d = data[:,3]

    resx   = gpu_res(resol, tx, olen)

    xv     = LinRange(xv_d[1], xv_d[end], resx+1)

    if do_interp
        bed, surf = interp(bed_d, surf_d, xv_d, xv)
        println("- Interpolating original data (nxv=$(size(bed_d)[1])) on nxv=$(size(bed)[1]) grid")
    else
        bed, surf = bed_d, surf_d
        println("- Using original data (nxv=$(size(bed_d)[1])) grid")
    end

    return bed, surf, xv
end


"""
    preprocess(dat_file::String; do_interp::Bool=true, resol::Int=128, do_rotate::Bool=true, fact_ny::Int=4, do_nondim::Bool=true, olen::Int=1)

Preprocess input data for iceflow model.

# Arguments
- `dat_file::String`: input data file with 3 space-limited columns as `[x-coord  z-bedrock  z-surface]`
- `do_interp::Bool=true`: perform linear interpolation of data
- `resol::Int=128`: output resolution
- `do_rotate::Bool=true`: rotate domain to minimise non-used grid-points
- `fact_ny::Int=4`: grid-point increase in y-dim
- `do_nondim::Bool=true`: non-dimensionalise output
- `olen::Int=1`: overlength for arrays larger then `nx` and `ny`
"""
@views function preprocess(dat_file::String; do_interp::Bool=true, resol::Int=128, do_rotate::Bool=true, fact_ny::Int=4, do_nondim::Bool=true, olen::Int=1)

    println("Starting preprocessing ... ")

    # load the data
    ybed, ysurf, xv = load_data(dat_file, do_interp, resol, olen)

    y_avg   = 0.5 .* (ybed .+ ysurf)

    lin_fit, α = lsq_fit(y_avg, xv)

    if do_rotate
        ybedr, ysurfr = rotate(ybed, ysurf, lin_fit)
    else
        ybedr, ysurfr = ybed, ysurf
    end

    ymin   = minimum(ybedr)
    ybedr  = ybedr  .- ymin
    ysurfr = ysurfr .- ymin
    max∆y  = maximum(ysurf .- ybed)
    ly     = maximum(ysurfr)
    lx     = maximum(xv) - minimum(xv)
    xc     = 0.5(xv[1:end-1].+xv[2:end])
    nx     = length(xc)
    resy   = ceil(Int, ly/lx*nx)
    resy   = fact_ny * gpu_res(resy, ty, 0) - olen
    yv     = LinRange(0,ly,resy+1)
    yc     = 0.5(yv[1:end-1].+yv[2:end])
    ny     = length(yc)
    
    println("- Preprocessed data: nx=$nx, ny=$ny (dx=$(round(lx/nx, sigdigits=4)), dy=$(round(ly/ny, sigdigits=4)))")

    sc     = do_nondim ? 1.0/ly : 1.0
    inputs = InputParams2D(ybedr*sc, av(ybedr)*sc, ysurfr*sc, av(ysurfr)*sc, xv*sc, yv*sc, xc*sc, yc*sc, lx*sc, max∆y*sc, nx, ny, α)
    
    println("... done.")
    return inputs, y_avg, lin_fit
end

