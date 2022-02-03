using Statistics, GeoArrays, Interpolations, LinearAlgebra, MAT, PyPlot

const USE_GPU = haskey(ENV, "USE_GPU") ? parse(Bool, ENV["USE_GPU"]) : false
const gpu_id  = haskey(ENV, "GPU_ID" ) ? parse(Int , ENV["GPU_ID" ]) : 0
###
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
    CUDA.device!(gpu_id)
else
    @init_parallel_stencil(Threads, Float64, 3)
end
const air   = 0.0
const fluid = 1.0
const solid = 2.0

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
function my_filter(A, mask)
    return [A[i] for i in eachindex(A) if mask[i] != 0]
end


"""
    lsq_fit(mask, zavg, xv2, yv2)

Linear least-square fit of mean bedrock and surface data.
"""
@views function lsq_fit(mask, zavg, xv2, yv2)
    # remove masked points
    xv2_    = my_filter(xv2 , mask)#[xv2[i]  for i in eachindex(xv2)  if mask[i] != 0]
    yv2_    = my_filter(yv2 , mask)#[yv2[i]  for i in eachindex(yv2)  if mask[i] != 0]
    zavg_   = my_filter(zavg, mask)#[zavg[i] for i in eachindex(zavg) if mask[i] != 0]
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


"Rotate field `A`, `x2v`, `y2v` with rotation matrix `R`."
function my_rot(R, X, Y, Z)
    xrot = R[1,1].*X .+ R[1,2].*Y .+ R[1,3].*Z
    yrot = R[2,1].*X .+ R[2,2].*Y .+ R[2,3].*Z
    zrot = R[3,1].*X .+ R[3,2].*Y .+ R[3,3].*Z
    return xrot, yrot, zrot
end


"Rotate field `A`, `x2v`, `y2v` with rotation matrix `R` and return extent."
function my_rot_minmax(R, X, Y, Z)
    xrot, yrot, zrot = my_rot(R, X, Y, Z)
    xmin,xmax = minimum(xrot), maximum(xrot)
    ymin,ymax = minimum(yrot), maximum(yrot)
    zmin,zmax = minimum(zrot), maximum(zrot)
    return xmin, xmax, ymin, ymax, zmin, zmax
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
@views function preprocess(dat_name::String; resx::Int=128, resy::Int=128, do_rotate::Bool=true, do_nondim::Bool=true, fact_nz::Int=2, olen::Int=1)
    
    println("Starting preprocessing ... ")

    println("- load the data")
    file1  = ("../data/alps/IceThick_cr0_$(dat_name).tif")
    file2  = ("../data/alps/BedElev_cr_$(dat_name).tif"  )
    zthick = reverse(GeoArrays.read(file1)[:,:,1], dims=2)
    zbed   = reverse(GeoArrays.read(file2)[:,:,1], dims=2)
    coord  = reverse(GeoArrays.coords(GeoArrays.read(file2)), dims=2)  

    # DEBUG: a step here could be rotation of the (x,y) plane using bounding box (rotating calipers)
    
    # coords and centre data in x,y plane
    (x2v,y2v)  = (getindex.(coord,1), getindex.(coord,2))
    xmin, xmax = extrema(x2v)
    ymin, ymax = extrema(y2v)
    ∆x, ∆y     = xmax-xmin, ymax-ymin
    x2v       .= x2v .- xmin .- ∆x/2
    y2v       .= y2v .- ymin .- ∆y/2

    # define and apply masks
    mask = ones(size(zthick))
    mask[ismissing.(zthick)] .= 0

    zthick[mask.==0] .= 0
    zthick = convert(Matrix{Float64}, zthick)

    zbed[ismissing.(zbed)] .= mean(my_filter(zbed,mask))
    zbed = convert(Matrix{Float64}, zbed)

    # define surface and avg topo
    zsurf = zbed .+ zthick
    zavg  = 0.5(zsurf .+ zbed)

    # retrieve extrema and centre data
    zmin, zmax = minimum( my_filter(zbed,mask) ),maximum( my_filter(zsurf,mask) )
    ∆z         = zmax-zmin
    zbed      .= zbed  .- zmin
    zsurf     .= zsurf .- zmin
    zavg      .= zavg  .- zmin

    if do_rotate
        println("- perform least square fit")
        plane, x = lsq_fit(mask, zavg, x2v, y2v)
        αx, αy, ori = x[1], x[2], x[3]
    else
        αx, αy, ori = 0.0, 0.0, 0.0
    end
 
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

    # DEBUG: one could here stop and export an HDF5 file including data, x,y coords, rotation matrix and ori
    # ----------------
    # DEBUG: since here, it could be done in code
    # rotate surface
    xsmin, xsmax, ysmin, ysmax, zsmin, zsmax = my_rot_minmax(R, x2v, y2v, zsurf)

    # rotate bed
    xbmin, xbmax, ybmin, ybmax, zbmin, zbmax = my_rot_minmax(R, x2v, y2v, zbed)

    # get extents
    xrmin,xrmax = min(xsmin,xbmin), max(xsmax,xbmax)
    yrmin,yrmax = min(ysmin,ybmin), max(ysmax,ybmax)
    zrmin,zrmax = zbmin, zbmax
    lx,ly,lz    = xrmax-xrmin, yrmax-yrmin, zrmax-zrmin

    # GPU friendly resolution nx, ny, nz
    nx =           gpu_res(resx               , tx, olen)
    ny =           gpu_res(resy               , ty, olen)
    nz = fact_nz * gpu_res(ceil(Int, lz/lx*nx), tz, 0   ) - olen
    println("- define nx, ny, nz = $nx, $ny, $nz (tx=$tx, ty=$ty, tz=$tz)")

    xv_d, yv_d = x2v[:,1], y2v[1,:]
    xv = LinRange(xv_d[1], xv_d[end], nx+1)
    yv = LinRange(yv_d[1], yv_d[end], ny+1)

    zbed2, zthick2 = interp(zbed, zthick, xv_d, yv_d, xv, yv)
    println("- interpolate original data (nxv, nyv = $(size(zbed)[1]), $(size(zbed)[2])) on nxv, nyv = $(size(zbed2)[1]), $(size(zbed2)[2]) grid")

    nsmb = 0#ceil(Int,nx/20)
    nsmt = 0#ceil(Int,nx/20)
    println("- apply smoothing ($nsmb steps on bed, $nsmt steps on thickness)")
    for ismb=1:nsmb
        smooth2D!(zbed2, 1.0)
    end
    for ismt=1:nsmt
        smooth2D!(zthick2, 1.0)
    end

    # reconstruct surface
    zsurf2 = zbed2 .+ zthick2

    # display(heatmap(xv, yv, zthick2'))
    # error("stop")

    # 3D grid
    xc, yc, zc = LinRange(xrmin-0.01lx,xrmax+0.01lx,nx), LinRange(yrmin-0.01ly,yrmax+0.01ly,ny), LinRange(zrmin-0.01lz,zrmax+0.01lz,nz)
    dx, dy, dz = xc[2]-xc[1], yc[2]-yc[1], zc[2]-zc[1]
    (x3,y3,z3) = ([x for x=xc,y=yc,z=zc], [y for x=xc,y=yc,z=zc], [z for x=xc,y=yc,z=zc])

    # rotate grid
    Rinv   = R'
    x3rot, y3rot, z3rot = my_rot(Rinv, x3, y3, z3)
    
    # set phases
    println("- set phases (0-air, 1-ice, 2-bedrock)")
    ϕ      = air .* @ones(size(x3))
    x3rot  = Data.Array(x3rot)
    y3rot  = Data.Array(y3rot)
    z3rot  = Data.Array(z3rot)
    zsurf2 = Data.Array(zsurf2)
    zbed2  = Data.Array(zbed2)
    @parallel set_phases!(ϕ, x3rot, y3rot, z3rot, zsurf2, zbed2, -xrmax, -yrmax, dx, dy)
    
    sc     = do_nondim ? 1.0/lz : 1.0    
    inputs = InputParams3D(ϕ, x3rot*sc, y3rot*sc, z3rot*sc, x3*sc, y3*sc, z3*sc, xc*sc, yc*sc, zc*sc, R, lx*sc, ly*sc, lz*sc, nx, ny, nz, sc)

    # vv = copy(zbrot);  vv[mask.==0] .=NaN
    sl = copy(ϕ)
    # p1 = heatmap(sl[:,1,:]')
    # p2 = heatmap(sl[65,:,:]')
    # p3 = heatmap(sl[:,85,:]')
    # p4 = heatmap(sl[:,115,:]')
    # p2 = heatmap(sl[Int(ceil(size(sl,1)/3)),:,:]')
    # display(plot(p1,p2, p3, p4))
    clf()
    subplot(311),pcolor(x3rot[:,ceil(Int,  ny/4),:], z3rot[:,ceil(Int,  ny/4),:], sl[:,ceil(Int,  ny/4),:]),colorbar()
    subplot(312),pcolor(x3rot[:,ceil(Int,  ny/2),:], z3rot[:,ceil(Int,  ny/2),:], sl[:,ceil(Int,  ny/2),:]),colorbar()
    subplot(313),pcolor(x3rot[:,ceil(Int,3*ny/4),:], z3rot[:,ceil(Int,3*ny/4),:], sl[:,ceil(Int,3*ny/4),:]),colorbar()
    
    # !ispath("../out_visu") && mkdir("../out_visu")
    # matwrite("../out_visu/out_pa3D.mat",
    #     Dict("Phase"=> Array(ϕ),
    #          "x3rot"=> Array(x3rot*sc), "y3rot"=> Array(y3rot*sc), "z3rot"=> Array(z3rot*sc),
    #          "x3"=> Array(x3*sc), "y3"=> Array(y3*sc), "z3"=> Array(z3*sc),
    #          "xc"=> Array(xc*sc), "yc"=> Array(yc*sc), "zc"=> Array(zc*sc),
    #          "lx"=> lx, "ly"=> ly, "lz"=> lz, "sc"=> sc); compress = true)

    println("... done.")
    return inputs
end

# preprocessing
inputs = preprocess("Rhone"; resx=256, resy=256, do_rotate=true, fact_nz=1)
