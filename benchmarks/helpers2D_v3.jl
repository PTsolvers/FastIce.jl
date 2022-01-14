using DelimitedFiles, Interpolations#, PyPlot

"Heuristic number of threads per block in x-dim."
const tx = 32

"Heuristic number of threads per block in y-dim."
const ty = 8

"Input parameters structure"
mutable struct InputParams2D
    ϕ; x2rot; y2rot; x2; y2; xc; yc; lx; ly; nx::Int; ny::Int; α; sc;
end


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
    α       = atan(sum((xv .- x_mean) .* (y_avg .- y_mean)) ./ sum((xv .- x_mean).^2))
    orig    = y_mean .- α .* x_mean
    lin_fit = α.*xv .+ orig
    return lin_fit, α
end

"Check if index is inside phase."
function is_inside_phase(y2rot,ytopo)
    return y2rot < ytopo
end


"""
    set_phases!(ϕ, x2rot, y2rot, ysurf, ybed, ox, dx)

Define phases as function of surface and bad topo.
"""
@parallel_indices (ix,iy) function set_phases!(ϕ, x2rot, y2rot, ysurf, ybed, ox, dx)
    if checkbounds(Bool,ϕ,ix,iy)
        ixr = clamp(floor(Int, (x2rot[ix,iy]-ox)/dx) + 1, 1, length(ysurf))
        if is_inside_phase(y2rot[ix,iy],ysurf[ixr])
            ϕ[ix,iy] = fluid
        end
        if is_inside_phase(y2rot[ix,iy],ybed[ixr])
            ϕ[ix,iy] = solid
        end
    end
    return
end


"Apply one explicit diffusion step as smoothing"
@parallel function smooth!(A2, A, fact)
    @inn(A2) = @inn(A) + 1.0/4.1/fact*(@d2_xi(A) + @d2_yi(A))
    return
end


"Round phases after applying smoothing."
@parallel_indices (ix,iy) function round_phase!(A)
    if checkbounds(Bool,A,ix,iy)
        if (A[ix,iy] <= 0.9                   ) A[ix,iy] = air   end
        if (A[ix,iy] >  0.9 && A[ix,iy] <= 1.9) A[ix,iy] = fluid end
        if (A[ix,iy] >  1.9                   ) A[ix,iy] = solid end
    end
    return
end


"""
    preprocess(dat_file::String; do_interp::Bool=true, resol::Int=128, do_rotate::Bool=true, fact_ny::Int=4, do_nondim::Bool=true, olen::Int=1)

Preprocess input data for iceflow model.

# Arguments
- `dat_file::String`: input data file with 3 space-limited columns as `[x-coord  z-bedrock  z-surface]`
- `resx::Int=128`: output x-dim resolution
- `do_rotate::Bool=true`: rotate domain to minimise non-used grid-points
- `do_nondim::Bool=true`: non-dimensionalise output
- `fact_ny::Int=4`: grid-point increase in y-dim
- `olen::Int=1`: overlength for arrays larger then `nx` and `ny`
"""
@views function preprocess(dat_file::String; resx::Int=128, do_rotate::Bool=true, do_nondim=true, fact_ny::Int=4, olen::Int=1)

    println("Starting preprocessing ... ")

    # load the data
    println("- load the data")
    data    = readdlm(dat_file, Float64)
    xv_d    = data[:,1]
    ybed_d  = data[:,2]
    ysurf_d = data[:,3]

    # GPU friendly resolution nx
    nx = gpu_res(resx, tx, olen)
    xv = LinRange(xv_d[1], xv_d[end], nx+1)

    ybed, ysurf = interp(ybed_d, ysurf_d, xv_d, xv)
    println("- interpolate original data (nxv=$(size(ybed_d)[1])) on nxv=$(size(ybed)[1]) grid")

    if do_rotate
        println("- perform least square fit")
        y_avg = 0.5 .* (ybed .+ ysurf)
        lin_fit, α  = lsq_fit(y_avg, xv)
        α = -α
    else
        α = 0.0
    end

    # retrieve extrema
    dx          = xv[2] - xv[1]
    xmin,xmax   = extrema(xv)
    ymin,ymax   = minimum(ybed),maximum(ysurf)
    ox,oy       = -(xmax-xmin)/2,-(ymax-ymin)/2

    # center data
    ybed        = ybed  .- ymin .+ oy
    ysurf       = ysurf .- ymin .+ oy
    xv          = xv    .- xmin .+ ox

    # rotate surface
    xsrot       = xv*cos(α) .- ysurf*sin(α)
    ysrot       = xv*sin(α) .+ ysurf*cos(α)
    xsmin,xsmax = minimum(xsrot), maximum(xsrot)
    ysmin,ysmax = minimum(ysrot), maximum(ysrot)

    # rotate bed
    xbrot       = xv*cos(α) .- ybed*sin(α)
    ybrot       = xv*sin(α) .+ ybed*cos(α)
    xbmin,xbmax = minimum(xbrot), maximum(xbrot)
    ybmin,ybmax = minimum(ybrot), maximum(ybrot)

    # get extents
    xrmin,xrmax = min(xsmin,xbmin),max(xsmax,xbmax)
    yrmin,yrmax = min(ysmin,ybmin),max(ysmax,ybmax)
    lx,ly       = xrmax-xrmin, yrmax-yrmin

    # GPU friendly resolution ny
    ny      = fact_ny * gpu_res(ceil(Int, ly/lx*nx), ty, 0) - olen
    println("- define nx, ny = $nx, $ny (tx=$tx, ty=$ty)")

    # 2D grid
    xc, yc  = LinRange(xrmin-0.01lx,xrmax+0.01lx,nx), LinRange(yrmin-0.01ly,yrmax+0.01ly,ny)
    (x2,y2) = ([x for x=xc,y=yc], [y for x=xc,y=yc])

    # rotate grid
    x2rot   =   x2*cos(α) .+ y2*sin(α)
    y2rot   = .-x2*sin(α) .+ y2*cos(α)
    
    # set phases
    println("- set phases (0-air, 1-ice, 2-bedrock)")
    ϕ      = air .* @ones(size(x2))
    x2rot  = Data.Array(x2rot)
    y2rot  = Data.Array(y2rot)
    ysurf  = Data.Array(ysurf)
    ybed   = Data.Array(ybed)
    @parallel set_phases!(ϕ, x2rot, y2rot, ysurf, ybed, ox, dx)

    nsm = ceil(Int,nx/50)
    println("- apply smoothing ($nsm diffusion steps)")
    ϕ2 = copy(ϕ)    
    for ism=1:nsm
        @parallel smooth!(ϕ2, ϕ, 1.0)
        ϕ, ϕ2 = ϕ2, ϕ
    end
    @parallel round_phase!(ϕ)

    sc     = do_nondim ? 1.0/ly : 1.0
    inputs = InputParams2D(ϕ, x2rot*sc, y2rot*sc, x2*sc, y2*sc, xc*sc, yc*sc, lx*sc, ly*sc, nx, ny, α, sc)

    # clf()
    # subplot(211), pcolor(x2rot*sc,y2rot*sc,Array(ϕ))
    # subplot(212), pcolor(x2*sc,y2*sc,Array(ϕ))
    # axis("image")

    println("... done.")
    return inputs
end
