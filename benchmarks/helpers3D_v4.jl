using Statistics, GeoArrays, Interpolations, LinearAlgebra#, MAT, PyPlot

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
@parallel_indices (ix,iy,iz) function set_phases!(ϕ, x3rot, y3rot, z3rot, zsurf, zbed, ox, oy, dx, dy, ns)
    if checkbounds(Bool,ϕ,ix,iy,iz)
        ixr = clamp(floor(Int, (x3rot[ix,iy,iz]-ox)/dx*ns) + 1, 1, size(zsurf,1))
        iyr = clamp(floor(Int, (y3rot[ix,iy,iz]-oy)/dy*ns) + 1, 1, size(zsurf,2))
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
@views function smooth2D!(A, B, fact)
    A[2:end-1,2:end-1] .= B[2:end-1,2:end-1] .+ 1.0/4.1/fact*(diff(diff(B[:,2:end-1],dims=1),dims=1) .+ diff(diff(B[2:end-1,:],dims=2),dims=2))
    return
end


# "Apply one explicit diffusion step as smoothing"
# @parallel function smooth!(A, B, fact)
#     @inn(A) = @inn(B) + 1.0/6.1/fact*(@d2_xi(B) + @d2_yi(B) + @d2_zi(B))
#     return
# end


# "Round phases after applying smoothing."
# @parallel_indices (ix,iy,iz) function round_phase!(A, B)
#     if checkbounds(Bool,A,ix,iy,iz)
#         if (A[ix,iy,iz] <= 0.5                      ) A[ix,iy,iz] = air   end
#         if (A[ix,iy,iz] >  0.5 && A[ix,iy,iz] <= 1.5) A[ix,iy,iz] = fluid end
#         if (A[ix,iy,iz] >  0.5 && A[ix,iy,iz] <= 1.5 && B[ix,iy,iz] ≠ fluid) A[ix,iy,iz] = air end
#         if (A[ix,iy,iz] >  1.5                      ) A[ix,iy,iz] = solid end
#     end
#     return
# end


"""
    preprocess1(dat_file::String; do_rotate::Bool=true)

Preprocess input data for iceflow model.

# Arguments
- `dat_file::String`: input data file
- `do_rotate=true`: perform data rotation
"""
@views function preprocess1(dat_name::String; do_rotate::Bool=true)

    println("Starting preprocessing 1 ... ")

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

    println("... done.")
    # DEBUG: one could here stop and export an HDF5 file including data, x,y coords, rotation matrix and ori
    return zsurf, zbed, zthick, x2v, y2v, R, ori
end


"""
    preprocess2(zsurf, zbed, zthick, x2v, y2v, R, ori; resx::Int=128, resy::Int=128, do_nondim::Bool=true, fact_nz::Int=2, ns=::Int=4, olen::Int=1)

Preprocess input data for iceflow model.

# Arguments
- `zsurf`: 2D surface elevation data
- `zbed`: 2D bedrock elevation data
- `ice`: 2D ice thickness data
- `x2v`, `y2v`: 2D x-y coords
- `R`: rotation matrix
- `ori`: rotation centre
- `resx::Int=128`: output x-resolution
- `resy::Int=128`: output y-resolution
- `do_nondim=true`: perform non-dimensionalisation
- `fact_nz::Int=2`: grid-point increase in z-dim
- `ns=::Int=4`: number of oversampling to limit aliasing
- `olen::Int=1`: overlength for arrays larger then `nx`, `ny` and `nz`
"""
@views function preprocess2(zsurf, zbed, zthick, x2v, y2v, R, ori; resx::Int=128, resy::Int=128, do_nondim::Bool=true, fact_nz::Int=2, ns::Int=4, olen::Int=1)
    # DEBUG: since here, it could be done in code
    println("Starting preprocessing 2 ... ")

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
    xv, yv = LinRange(xv_d[1], xv_d[end], ns*(nx+1)), LinRange(yv_d[1], yv_d[end], ns*(ny+1))

    zbed2, zthick2 = interp(zbed, zthick, xv_d, yv_d, xv, yv)
    println("- interpolate original data (nxv, nyv = $(size(zbed)[1]), $(size(zbed)[2])) on nxv, nyv = $(size(zbed2)[1]), $(size(zbed2)[2]) grid ($(ns)x oversampling)")
    
    nsmb, nsmt = 5, 5 #ceil(Int,nx/20)
    println("- apply smoothing ($nsmb steps on bed, $nsmt steps on thickness)")
    Tmp = copy(zbed2);   for ismb=1:nsmb smooth2D!(zbed2  , Tmp, 1.0)  end
    Tmp = copy(zthick2); for ismt=1:nsmt smooth2D!(zthick2, Tmp, 1.0)  end

    # reconstruct surface
    zsurf2 = zbed2 .+ zthick2

    # 3D grid
    xc, yc, zc = LinRange(xrmin-0.01lx,xrmax+0.01lx,nx), LinRange(yrmin-0.01ly,yrmax+0.01ly,ny), LinRange(zrmin-0.01lz,zrmax+0.01lz,nz)
    dx, dy, dz = xc[2]-xc[1], yc[2]-yc[1], zc[2]-zc[1]
    (x3,y3,z3) = ([x for x=xc,y=yc,z=zc], [y for x=xc,y=yc,z=zc], [z for x=xc,y=yc,z=zc])

    # rotate grid
    Rinv = R'
    x3rot, y3rot, z3rot = my_rot(Rinv, x3, y3, z3)
    
    # set phases
    println("- set phases (0-air, 1-ice, 2-bedrock)")
    ϕ      = air .* @ones(size(x3))
    x3rot  = Data.Array(x3rot)
    y3rot  = Data.Array(y3rot)
    z3rot  = Data.Array(z3rot)
    zsurf2 = Data.Array(zsurf2)
    zbed2  = Data.Array(zbed2)
    @parallel set_phases!(ϕ, x3rot, y3rot, z3rot, zsurf2, zbed2, -xrmax, -yrmax, dx, dy, ns)
    
    # Tmp1 = copy(ϕ)
    # for ii=1:3
    # Tmp = copy(ϕ)
    # @parallel smooth!(ϕ, Tmp, 1.0)
    # @parallel round_phase!(ϕ, Tmp)
    # end
    
    sc     = do_nondim ? 1.0/lz : 1.0    
    inputs = InputParams3D(ϕ, x3rot*sc, y3rot*sc, z3rot*sc, x3*sc, y3*sc, z3*sc, xc*sc, yc*sc, zc*sc, R, lx*sc, ly*sc, lz*sc, nx, ny, nz, sc)

    # sl = copy(ϕ)
    # clf()
    # subplot(311),pcolor(x3rot[:,ceil(Int,  ny/4),:], z3rot[:,ceil(Int,  ny/4),:], sl[:,ceil(Int,  ny/4),:]),colorbar()
    # subplot(312),pcolor(x3rot[:,ceil(Int,  ny/2),:], z3rot[:,ceil(Int,  ny/2),:], sl[:,ceil(Int,  ny/2),:]),colorbar()
    # subplot(313),pcolor(x3rot[:,ceil(Int,3*ny/4),:], z3rot[:,ceil(Int,3*ny/4),:], sl[:,ceil(Int,3*ny/4),:]),colorbar()
    
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
# zsurf, zbed, zthick, x2v, y2v, R, ori = preprocess1("Rhone"; do_rotate=true)
# inputs = preprocess2(zsurf, zbed, zthick, x2v, y2v, R, ori; resx=128, resy=128, fact_nz=2, ns=16)
