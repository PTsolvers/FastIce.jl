using DelimitedFiles, Interpolations, MAT, LinearAlgebra#, PyPlot

# const USE_GPU = haskey(ENV, "USE_GPU") ? parse(Bool, ENV["USE_GPU"]) : false
# const gpu_id  = haskey(ENV, "GPU_ID" ) ? parse(Int , ENV["GPU_ID" ]) : 0
# ###
# using ParallelStencil
# using ParallelStencil.FiniteDifferences3D
# @static if USE_GPU
#     @init_parallel_stencil(CUDA, Float64, 3)
#     CUDA.device!(gpu_id)
# else
#     @init_parallel_stencil(Threads, Float64, 3)
# end
# const air   = 0.0
# const fluid = 1.0
# const solid = 2.0

"Heuristic number of threads per block in x-dim."
const tx = 32

"Heuristic number of threads per block in y-dim."
const ty = 8

"Heuristic number of threads per block in z-dim."
const tz = 8

"Input parameters structure"
mutable struct InputParams3D
    ϕ; x3rot; y3rot; z3rot; x3; y3; z3; xc; yc; zc; R; lx; ly; lz; nx::Int; ny::Int; nz::Int; sc;
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
    interp(A, B, xv_d, yv_d, xv, yv)

Linear interpolation of fields `A` and `B` on `(xv, yv)` grid.
"""
@views function interp(A, B, xv_d, yv_d, xv, yv)

    itp1 = interpolate( (xv_d,yv_d), A, Gridded(Linear()))
    itp2 = interpolate( (xv_d,yv_d), B, Gridded(Linear()))
    
    A2   = [itp1(x,y) for x in xv, y in yv]
    B2   = [itp2(x,y) for x in xv, y in yv]
    
    @assert length(xv) == size(A2)[1] == size(A2)[1]
    @assert length(yv) == size(A2)[2] == size(A2)[2]
    return A2, B2
end


"Filter out all values of `A` based on `mask`."
function filter(A, mask)
    return [A[i] for i in eachindex(A) if mask[i] != 0]
end

"""
    lsq_fit(mask, zavg, xv2, yv2)

Linear least-square fit of mean bedrock and surface data.
"""
@views function lsq_fit(mask, zavg, xv2, yv2)
    # remove masked points
    xv2_    = filter(xv2 , mask)#[xv2[i]  for i in eachindex(xv2)  if mask[i] != 0]
    yv2_    = filter(yv2 , mask)#[yv2[i]  for i in eachindex(yv2)  if mask[i] != 0]
    zavg_   = filter(zavg, mask)#[zavg[i] for i in eachindex(zavg) if mask[i] != 0]
    # prepare least square (lsq)
    A       =  ones(length(xv2_[:]),3)
    B       = zeros(length(xv2_[:]),1)
    A[:,1] .= xv2_[:]
    A[:,2] .= yv2_[:]
    B      .= zavg_[:]
    # lsq solve x (x-slope, y-slope, origin)
    x       = (A'*A) \ (A'*B)
    # best fitting plane
    plane   = xv2*x[1] .+ yv2*x[2] .+ x[3];
    return plane, x
end


"Check if index is inside phase."
function is_inside_phase(z3rot,ztopo)
    return z3rot < ztopo
end


"""
    set_phases!(ϕ, x3rot, y3rot, z3rot, zsurf, zbed, ox, oy, dx, dy)

Define phases as function of surface and bad topo.
"""
@parallel_indices (ix,iy,iz) function set_phases!(ϕ, x3rot, y3rot, z3rot, zsurf, zbed, ox, oy, dx, dy)
    if checkbounds(Bool,ϕ,ix,iy,iz)
        ixr = clamp(floor(Int, (x3rot[ix,iy,iz]-ox)/dx) + 1, 1, size(zsurf,1))
        iyr = clamp(floor(Int, (y3rot[ix,iy,iz]-oy)/dy) + 1, 1, size(zsurf,2))
        if is_inside_phase(z3rot[ix,iy,iz],zsurf[ixr,iyr])
            ϕ[ix,iy,iz] = fluid
        end
        if is_inside_phase(z3rot[ix,iy,iz],zbed[ixr,iyr])
            ϕ[ix,iy,iz] = solid
        end
    end
    return
end


"Apply one explicit diffusion step as smoothing"
@views function smooth2D!(A, fact)
    A[2:end-1,2:end-1] .= A[2:end-1,2:end-1] .+ 1.0/4.1/fact*(diff(diff(A[:,2:end-1],dims=1),dims=1) .+ diff(diff(A[2:end-1,:],dims=2),dims=2))
    return
end


# "Round phases after applying smoothing."
# @parallel_indices (ix,iy,iz) function round_phase!(A)
#     if checkbounds(Bool,A,ix,iy,iz)
#         if (A[ix,iy,iz] <= 0.9                      ) A[ix,iy,iz] = air   end
#         if (A[ix,iy,iz] >  0.9 && A[ix,iy,iz] <= 1.9) A[ix,iy,iz] = fluid end
#         if (A[ix,iy,iz] >  1.9                      ) A[ix,iy,iz] = solid end
#     end
#     return
# end


"""
    preprocess(dat_file::String; resx::Int=128, resy::Int=128, do_rotate::Bool=true, do_nondim::Bool=true, fact_nz::Int=2, olen::Int=1)

Preprocess input data for iceflow model.

# Arguments
- `dat_file::String`: input data file
- `resx::Int=128`: output x-resolution
- `resy::Int=128`: output y-resolution
- `do_rotate=true`: perform data rotation
- `do_nondim=true`: perform non-dimensionalisation
- `fact_nz::Int=2`: grid-point increase in z-dim
- `olen::Int=1`: overlength for arrays larger then `nx`, `ny` and `nz`
"""
@views function preprocess(dat_file::String; resx::Int=128, resy::Int=128, do_rotate::Bool=true, do_nondim::Bool=true, fact_nz::Int=2, olen::Int=1)
    
    println("Starting preprocessing ... ")

    # load the data
    println("- load the data")
    vars     = matread(dat_file)
    zbed_d   = get(vars,"zbed"  ,1)
    zthick_d = get(vars,"zthick",1)
    xv_d     = get(vars,"xv"   ,1); xv_d = vec(Float64.(xv_d))
    yv_d     = get(vars,"yv"   ,1); yv_d = vec(Float64.(yv_d))

    # GPU friendly resolution nx, ny
    nx = gpu_res(resx, tx, olen)
    ny = gpu_res(resy, ty, olen)
    xv = LinRange(xv_d[1], xv_d[end], nx+1)
    yv = LinRange(yv_d[1], yv_d[end], ny+1)

    zbed, zthick = interp(zbed_d, zthick_d, xv_d, yv_d, xv, yv)
    println("- interpolate original data (nxv, nyv = $(size(zbed_d)[1]), $(size(zbed_d)[2])) on nxv, nyv = $(size(zbed)[1]), $(size(zbed)[2]) grid")

    nsmb = 5#ceil(Int,nx/20)
    nsmt = 10#ceil(Int,nx/20)
    println("- apply smoothing ($nsmb steps on bed, $nsmt steps on thickness)")
    for ismb=1:nsmb
        smooth2D!(zbed, 1.0)
    end
    for ismt=1:nsmt
        smooth2D!(zthick, 1.0)
    end

    # define surface
    zsurf = zbed .+ zthick

    # define mask
    mask = ones(size(zthick))
    mask[zthick.==0] .= 0

    (x2v,y2v) = ([x for x=xv,y=yv], [y for x=xv,y=yv])
    zavg = 0.5(zsurf .+ zbed)

    if do_rotate
        println("- perform least square fit")
        plane, x = lsq_fit(mask, zavg, x2v, y2v)
        αx, αy, ori = x[1], x[2], x[3]
    else
        αx, αy, ori = 0.0, 0.0, 0.0
    end

    # retrieve extrema
    dx          = xv[2] - xv[1]
    dy          = yv[2] - yv[1]
    xmin,xmax   = extrema(xv)
    ymin,ymax   = extrema(yv)
    zmin,zmax   = minimum(zbed),maximum(zsurf)
    if do_rotate
        ox,oy,oz = -(xmax-xmin)/2,-(ymax-ymin)/2,-ori + zmin
    else
        ox,oy,oz = -(xmax-xmin)/2,-(ymax-ymin)/2,-(zmax-zmin)/2
    end

    # center data
    zbed        = zbed  .- zmin .+ oz
    zsurf       = zsurf .- zmin .+ oz
    xv          = xv    .- xmin .+ ox
    yv          = yv    .- ymin .+ oy
    (x2v,y2v)   = ([x for x=xv,y=yv], [y for x=xv,y=yv])

    nv          = [-αx -αy 1.0]
    nv        ./= norm(nv)
    if do_rotate
        ax      = [nv[2] -nv[1] 0.0]
        ax    ./= norm(ax)
    else
        ax      = [0.0 0.0 0.0]
    end
    θ           = acos(nv[3])

    R = [cos(θ)+ax[1]^2*(1-cos(θ))  ax[1]*ax[2]*(1-cos(θ))       ax[2]*sin(θ)
         ax[2]*ax[1]*(1-cos(θ))     cos(θ) + ax[2]^2*(1-cos(θ)) -ax[1]*sin(θ)
        -ax[2]*sin(θ)               ax[1]*sin(θ)                       cos(θ)]

    # rotate surface
    xsrot       =   R[1,1]*x2v .+ R[1,2]*y2v .+ R[1,3]*zsurf
    ysrot       =   R[2,1]*x2v .+ R[2,2]*y2v .+ R[2,3]*zsurf
    zsrot       =   R[3,1]*x2v .+ R[3,2]*y2v .+ R[3,3]*zsurf
    xsmin,xsmax = minimum(xsrot), maximum(xsrot)
    ysmin,ysmax = minimum(ysrot), maximum(ysrot)
    zsmin,zsmax = minimum(zsrot), maximum(zsrot)

    # rotate bed
    xbrot       =   R[1,1]*x2v .+ R[1,2]*y2v .+ R[1,3]*zbed
    ybrot       =   R[2,1]*x2v .+ R[2,2]*y2v .+ R[2,3]*zbed
    zbrot       =   R[3,1]*x2v .+ R[3,2]*y2v .+ R[3,3]*zbed
    xbmin,xbmax = minimum(xbrot), maximum(xbrot)
    ybmin,ybmax = minimum(ybrot), maximum(ybrot)
    zbmin,zbmax = minimum(zbrot), maximum(zbrot)

    # get extents
    xrmin,xrmax = min(xsmin,xbmin),max(xsmax,xbmax)
    yrmin,yrmax = min(ysmin,ybmin),max(ysmax,ybmax)
    zrmin,zrmax = min(zsmin,zbmin),max(zsmax,zbmax)
    lx,ly,lz    = xrmax-xrmin, yrmax-yrmin, zrmax-zrmin
    
    # GPU friendly resolution ny
    nz      = fact_nz * gpu_res(ceil(Int, lz/lx*nx), tz, 0) - olen
    println("- define nx, ny, nz = $nx, $ny, $nz (tx=$tx, ty=$ty, tz=$tz)")

    # 3D grid
    xc, yc, zc  = LinRange(xrmin-0.01lx,xrmax+0.01lx,nx), LinRange(yrmin-0.01ly,yrmax+0.01ly,ny), LinRange(zrmin-0.01lz,zrmax+0.01lz,nz)
    (x3,y3,z3)  = ([x for x=xc,y=yc,z=zc], [y for x=xc,y=yc,z=zc], [z for x=xc,y=yc,z=zc])

    Rinv    = R'
    # rotate grid
    x3rot   =  Rinv[1,1]*x3 .+ Rinv[1,2]*y3 .+ Rinv[1,3]*z3
    y3rot   =  Rinv[2,1]*x3 .+ Rinv[2,2]*y3 .+ Rinv[2,3]*z3
    z3rot   =  Rinv[3,1]*x3 .+ Rinv[3,2]*y3 .+ Rinv[3,3]*z3
    
    # set phases
    println("- set phases (0-air, 1-ice, 2-bedrock)")
    ϕ      = air .* @ones(size(x3))
    x3rot  = Data.Array(x3rot)
    y3rot  = Data.Array(y3rot)
    z3rot  = Data.Array(z3rot)
    zsurf  = Data.Array(zsurf)
    zbed   = Data.Array(zbed)
    @parallel set_phases!(ϕ, x3rot, y3rot, z3rot, zsurf, zbed, ox, oy, dx, dy)
    
    sc     = do_nondim ? 1.0/lz : 1.0    
    inputs = InputParams3D(ϕ, x3rot*sc, y3rot*sc, z3rot*sc, x3*sc, y3*sc, z3*sc, xc*sc, yc*sc, zc*sc, R, lx*sc, ly*sc, lz*sc, nx, ny, nz, sc)

    # vv = copy(zbrot);  vv[mask.==0] .=NaN
    # sl = copy(ϕ)
    # clf()
    # subplot(131),pcolor(vv'),colorbar()
    # # subplot(132),pcolor(sl[:,ceil(Int,ny/4),:]'),colorbar()
    # subplot(132),pcolor(x3rot[:,ceil(Int,ny/4),:], z3rot[:,ceil(Int,ny/4),:], sl[:,ceil(Int,ny/4),:]),colorbar()
    # # subplot(133),pcolor(sl[200,:,:]'),colorbar()
    # subplot(133),pcolor(y3rot[ceil(Int,3*nx/4),:,:], z3rot[ceil(Int,3*nx/4),:,:], sl[ceil(Int,3*nx/4),:,:]),colorbar()

    println("... done.")
    return inputs
end

# preprocessing
# inputs = preprocess("../data/arolla3D.mat"; resx=256, resy=256, do_rotate=true, fact_nz=2)

