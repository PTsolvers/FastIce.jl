const USE_GPU = haskey(ENV, "USE_GPU") ? parse(Bool, ENV["USE_GPU"]) : true
const gpu_id  = haskey(ENV, "GPU_ID" ) ? parse(Int , ENV["GPU_ID" ]) : 7
const do_save = haskey(ENV, "DO_SAVE") ? parse(Bool, ENV["DO_SAVE"]) : true
###
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end
using ImplicitGlobalGrid, Printf, Statistics, LinearAlgebra, Random, UnPack, Plots, HDF5, LightXML
import MPI

norm_g(A) = (sum2_l = sum(A.^2); sqrt(MPI.Allreduce(sum2_l, MPI.SUM, MPI.COMM_WORLD)))
sum_g(A)  = (sum_l  = sum(A); MPI.Allreduce(sum_l, MPI.SUM, MPI.COMM_WORLD))

@views inn(A) = A[2:end-1,2:end-1,2:end-1]

include(joinpath(@__DIR__, "helpers3D_v4_xpu.jl"))

import ParallelStencil: INDICES
ix,iy,iz    = INDICES[1], INDICES[2], INDICES[3]
ixi,iyi,izi = :($ix+1), :($iy+1), :($iz+1)

const air   = 0.0
const fluid = 1.0
const solid = 2.0

macro av_xii(A) esc(:( 0.5*($A[$ixi,$iyi,$izi] + $A[$ixi+1,$iyi  ,$izi  ]) )) end
macro av_yii(A) esc(:( 0.5*($A[$ixi,$iyi,$izi] + $A[$ixi  ,$iyi+1,$izi  ]) )) end
macro av_zii(A) esc(:( 0.5*($A[$ixi,$iyi,$izi] + $A[$ixi  ,$iyi  ,$izi+1]) )) end

macro fm(A)   esc(:( $A[$ix,$iy,$iz] == fluid )) end
macro fmxy(A) esc(:( !($A[$ix,$iy,$izi] == air || $A[$ix+1,$iy,$izi] == air || $A[$ix,$iy+1,$izi] == air || $A[$ix+1,$iy+1,$izi] == air) )) end
macro fmxz(A) esc(:( !($A[$ix,$iyi,$iz] == air || $A[$ix+1,$iyi,$iz] == air || $A[$ix,$iyi,$iz+1] == air || $A[$ix+1,$iyi,$iz+1] == air) )) end
macro fmyz(A) esc(:( !($A[$ixi,$iy,$iz] == air || $A[$ixi,$iy+1,$iz] == air || $A[$ixi,$iy,$iz+1] == air || $A[$ixi,$iy+1,$iz+1] == air) )) end

@parallel function compute_P_τ!(∇V, Pt, τxx, τyy, τzz, τxy, τxz, τyz, Vx, Vy, Vz, ϕ, r, μ_veτ, Gdτ, dx, dy, dz)
    @all(∇V)  = @fm(ϕ)*(@d_xa(Vx)/dx + @d_ya(Vy)/dy + @d_za(Vz)/dz)
    @all(Pt)  = @fm(ϕ)*(@all(Pt) - r*Gdτ*@all(∇V))    
    @all(τxx) = @fm(ϕ)*2.0*μ_veτ*(@d_xa(Vx)/dx + @all(τxx)/Gdτ/2.0)
    @all(τyy) = @fm(ϕ)*2.0*μ_veτ*(@d_ya(Vy)/dy + @all(τyy)/Gdτ/2.0)
    @all(τzz) = @fm(ϕ)*2.0*μ_veτ*(@d_za(Vz)/dz + @all(τzz)/Gdτ/2.0)
    @all(τxy) = @fmxy(ϕ)*2.0*μ_veτ*(0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx) + @all(τxy)/Gdτ/2.0)
    @all(τxz) = @fmxz(ϕ)*2.0*μ_veτ*(0.5*(@d_zi(Vx)/dz + @d_xi(Vz)/dx) + @all(τxz)/Gdτ/2.0)
    @all(τyz) = @fmyz(ϕ)*2.0*μ_veτ*(0.5*(@d_zi(Vy)/dz + @d_yi(Vz)/dy) + @all(τyz)/Gdτ/2.0)
    return
end

macro sm_xi(A) esc(:( !(($A[$ix,$iyi,$izi] == solid) || ($A[$ix+1,$iyi,$izi] == solid)) )) end
macro sm_yi(A) esc(:( !(($A[$ixi,$iy,$izi] == solid) || ($A[$ixi,$iy+1,$izi] == solid)) )) end
macro sm_zi(A) esc(:( !(($A[$ixi,$iyi,$iz] == solid) || ($A[$ixi,$iyi,$iz+1] == solid)) )) end

macro fm_xi(A) esc(:( ($A[$ix,$iyi,$izi] == fluid) && ($A[$ix+1,$iyi,$izi] == fluid) )) end
macro fm_yi(A) esc(:( ($A[$ixi,$iy,$izi] == fluid) && ($A[$ixi,$iy+1,$izi] == fluid) )) end
macro fm_zi(A) esc(:( ($A[$ixi,$iyi,$iz] == fluid) && ($A[$ixi,$iyi,$iz+1] == fluid) )) end

@parallel function compute_V!(Vx, Vy, Vz, Pt, τxx, τyy, τzz, τxy, τxz, τyz, ϕ, ρgx, ρgy, ρgz, dτ_ρ, dx, dy, dz)
    @inn(Vx) = @sm_xi(ϕ)*( @inn(Vx) + dτ_ρ*(@d_xi(τxx)/dx + @d_ya(τxy)/dy + @d_za(τxz)/dz - @d_xi(Pt)/dx - @fm_xi(ϕ)*ρgx) )
    @inn(Vy) = @sm_yi(ϕ)*( @inn(Vy) + dτ_ρ*(@d_yi(τyy)/dy + @d_xa(τxy)/dx + @d_za(τyz)/dz - @d_yi(Pt)/dy - @fm_yi(ϕ)*ρgy) )
    @inn(Vz) = @sm_zi(ϕ)*( @inn(Vz) + dτ_ρ*(@d_zi(τzz)/dy + @d_xa(τxz)/dx + @d_ya(τyz)/dy - @d_zi(Pt)/dz - @fm_zi(ϕ)*ρgz) )
    return
end

@parallel function compute_Res!(Rx, Ry, Rz, Pt, τxx, τyy, τzz, τxy, τxz, τyz, ϕ, ρgx, ρgy, ρgz, dx, dy, dz)
    @all(Rx)  = @sm_xi(ϕ)*(@d_xi(τxx)/dx + @d_ya(τxy)/dy + @d_za(τxz)/dz - @d_xi(Pt)/dx - @fm_xi(ϕ)*ρgx)
    @all(Ry)  = @sm_yi(ϕ)*(@d_yi(τyy)/dy + @d_xa(τxy)/dx + @d_za(τyz)/dz - @d_yi(Pt)/dy - @fm_yi(ϕ)*ρgy)
    @all(Rz)  = @sm_zi(ϕ)*(@d_zi(τzz)/dy + @d_xa(τxz)/dx + @d_ya(τyz)/dy - @d_zi(Pt)/dz - @fm_zi(ϕ)*ρgz)
    return
end

@parallel function preprocess_visu!(Vn, τII, Ptv, Vx, Vy, Vz, τxx, τyy, τzz, τxy, τxz, τyz, Pt)
    # all arrays of size (nx-2,ny-2,nz-2)
    @all(Vn)  = sqrt(@av_xii(Vx)*@av_xa(Vx) + @av_yii(Vy)*@av_yii(Vy) + @av_zii(Vz)*@av_zii(Vz))
    @all(τII) = sqrt(0.5*(@inn(τxx)*@inn(τxx) + @inn(τyy)*@inn(τyy) + @inn(τzz)*@inn(τzz)) + @av_xya(τxy)*@av_xya(τxy) + @av_xza(τxz)*@av_xza(τxz) + @av_yza(τyz)*@av_yza(τyz))
    @all(Ptv) = @inn(Pt)
    return
end

@parallel_indices (ix,iy,iz) function apply_mask!(Vn, τII, Ptv, ϕ)
    if checkbounds(Bool,Vn,ix,iy,iz)
        if ϕ[ix+1,iy+1,iz+1] != fluid
             Vn[ix,iy,iz] = NaN
            Ptv[ix,iy,iz] = NaN
            τII[ix,iy,iz] = NaN
        end
    end
    return
end

@parallel_indices (ix,iy,iz) function init_ϕi!(ϕ,ϕx,ϕy,ϕz)
    if ix <= size(ϕx,1) && iy <= size(ϕx,2) && iz <= size(ϕx,3)
        ϕx[ix,iy,iz] = air
        if ϕ[ix,iy,iz] == fluid && ϕ[ix+1,iy,iz] == fluid
            ϕx[ix,iy,iz] = fluid
        end
    end
    if ix <= size(ϕy,1) && iy <= size(ϕy,2) && iz <= size(ϕy,3)
        ϕy[ix,iy,iz] = air
        if ϕ[ix,iy,iz] == fluid && ϕ[ix,iy+1,iz] == fluid
            ϕy[ix,iy,iz] = fluid
        end
    end
    if ix <= size(ϕz,1) && iy <= size(ϕz,2) && iz <= size(ϕz,3)
        ϕz[ix,iy,iz] = air
        if ϕ[ix,iy,iz] == fluid && ϕ[ix,iy,iz+1] == fluid
            ϕz[ix,iy,iz] = fluid
        end
    end
    return
end

function create_xdmf_attribute(xgrid,file,name,dim_g)
    # TODO: solve type and precision
    xattr = new_child(xgrid, "Attribute")
    set_attribute(xattr, "Name", name)
    set_attribute(xattr, "Center", "Cell")
    xdata = new_child(xattr, "DataItem")
    set_attribute(xdata, "Format", "HDF")
    set_attribute(xdata, "NumberType", "Float")
    set_attribute(xdata, "Precision", "8")
    set_attribute(xdata, "Dimensions", join(reverse(dim_g), ' '))
    add_text(xdata, "$file:/$name")
    return xattr
end

@views function Stokes3D()
    # inputs
    filename  = "../data/alps/data_Rhone.h5"
    nx        = 511
    ny        = 511
    nz        = 383
    dim       = (2,2,2)
    
    do_nondim = true
    ns        = 4

    me, dims, nprocs, coords, comm_cart = init_global_grid(nx, ny, nz; dimx=dim[1], dimy=dim[2], dimz=dim[3]) # MPI initialisation

    if (me==0) println("Starting preprocessing ... ") end
    if (me==0) println("- read data from $(filename)") end
    fid    = h5open(filename, "r")
    zsurf  = read(fid,"glacier/zsurf")
    zbed   = read(fid,"glacier/zbed")
    zthick = read(fid,"glacier/zthick")
    x2v    = read(fid,"glacier/x2v")
    y2v    = read(fid,"glacier/z2v")
    R      = read(fid,"glacier/R")
    ori    = read(fid,"glacier/ori")
    close(fid)
    # rotate surface
    xsmin, xsmax, ysmin, ysmax, zsmin, zsmax = my_rot_minmax(R, x2v, y2v, zsurf)
    # rotate bed
    xbmin, xbmax, ybmin, ybmax, zbmin, zbmax = my_rot_minmax(R, x2v, y2v, zbed)
    # get extents
    xrmin,xrmax = min(xsmin,xbmin), max(xsmax,xbmax)
    yrmin,yrmax = min(ysmin,ybmin), max(ysmax,ybmax)
    zrmin,zrmax = zbmin, zbmax
    ∆x,∆y,∆z    = xrmax-xrmin, yrmax-yrmin, zrmax-zrmin
    # init global domain
    xc, yc, zc = LinRange(xrmin-0.01∆x,xrmax+0.01∆x,nx_g()), LinRange(yrmin-0.01∆y,yrmax+0.01∆y,ny_g()), LinRange(zrmin-0.01∆z,zrmax+0.01∆z,nz_g())
    dx, dy, dz = xc[2]-xc[1], yc[2]-yc[1], zc[2]-zc[1]
    lx, ly, lz = xc[end]-xc[1], yc[end]-yc[1], zc[end]-zc[1]

    # preprocessing
    xv_d, yv_d = x2v[:,1], y2v[1,:]
    xv, yv = LinRange(xv_d[1], xv_d[end], ns*(nx_g()+1)), LinRange(yv_d[1], yv_d[end], ns*(ny_g()+1))

    zbed2, zthick2 = interp(zbed, zthick, xv_d, yv_d, xv, yv)
    if (me==0) println("- interpolate original data (nxv, nyv = $(size(zbed)[1]), $(size(zbed)[2])) on nxv, nyv = $(size(zbed2)[1]), $(size(zbed2)[2]) grid ($(ns)x oversampling)") end
    
    nsmb, nsmt = 5, 5 #ceil(Int,nx/20)
    if (me==0) println("- apply smoothing ($nsmb steps on bed, $nsmt steps on thickness)") end
    Tmp = copy(zbed2);   for ismb=1:nsmb smooth2D!(zbed2  , Tmp, 1.0)  end
    Tmp = copy(zthick2); for ismt=1:nsmt smooth2D!(zthick2, Tmp, 1.0)  end

    # reconstruct surface
    zsurf2 = zbed2 .+ zthick2

    sc         = do_nondim ? 1.0/lz : 1.0
    # scaling
    xc, yc, zc = xc*sc, yc*sc, zc*sc
    dx, dy, dz = dx*sc, dy*sc, dz*sc
    lx, ly, lz = lx*sc, ly*sc, lz*sc
    xrmax, yrmax = xrmax*sc, yrmax*sc
    zsurf2     = zsurf2*sc
    zbed2      = zbed2*sc

    ox,oy,oz = -xrmax, -yrmax, ori

    # physics
    ## dimensionally independent
    μs0       = 1.0               # matrix viscosity [Pa*s]
    ρg0       = 1.0               # gravity          [Pa/m]
    ## scales
    psc       = ρg0*lz
    tsc       = μs0/psc
    vsc       = lz/tsc
    ## dimensionally dependent
    ρgv       = ρg0*R'*[0,0,1]
    ρgx,ρgy,ρgz = ρgv[1], ρgv[2], ρgv[3]
    # numerics
    maxiter   = 50nz_g()     # maximum number of pseudo-transient iterations
    nchk      = 2*nz_g()     # error checking frequency
    nviz      = 2*nz_g()     # visualisation frequency
    b_width   = (8,4,4)      # boundary width
    ε_V       = 1e-8         # nonlinear absolute tolerance for momentum
    ε_∇V      = 1e-8         # nonlinear absolute tolerance for divergence
    CFL       = 0.95/sqrt(3) # stability condition
    Re        = 2π           # Reynolds number                     (numerical parameter #1)
    r         = 1.0          # Bulk to shear elastic modulus ratio (numerical parameter #2)
    # preprocessing
    max_lxyz   = 0.25lz
    Vpdτ       = min(dx,dy,dz)*CFL
    dτ_ρ       = Vpdτ*max_lxyz/Re/μs0
    Gdτ        = Vpdτ^2/dτ_ρ/(r+2.0)
    μ_veτ      = 1.0/(1.0/Gdτ + 1.0/μs0)
    # allocation
    Pt        = @zeros(nx  ,ny  ,nz  )
    ∇V        = @zeros(nx  ,ny  ,nz  )
    τxx       = @zeros(nx  ,ny  ,nz  )
    τyy       = @zeros(nx  ,ny  ,nz  )
    τzz       = @zeros(nx  ,ny  ,nz  )
    τxy       = @zeros(nx-1,ny-1,nz-2)
    τxz       = @zeros(nx-1,ny-2,nz-1)
    τyz       = @zeros(nx-2,ny-1,nz-1)
    Rx        = @zeros(nx-1,ny-2,nz-2)
    Ry        = @zeros(nx-2,ny-1,nz-2)
    Rz        = @zeros(nx-2,ny-2,nz-1)
    ϕx        = @zeros(nx-1,ny-2,nz-2)
    ϕy        = @zeros(nx-2,ny-1,nz-2)
    ϕz        = @zeros(nx-2,ny-2,nz-1)
    Vx        = @zeros(nx+1,ny  ,nz  )
    Vy        = @zeros(nx  ,ny+1,nz  )
    Vz        = @zeros(nx  ,ny  ,nz+1)
    # set phases
    if (me==0) println("- set phases (0-air, 1-ice, 2-bedrock)") end
    Rinv     = Data.Array(R')
    zsurf_g  = Data.Array(zsurf2)
    zbed_g   = Data.Array(zbed2)
    ϕ        = air .* @ones(nx,ny,nz)
    @parallel set_phases!(ϕ, zsurf_g, zbed_g, Rinv, xc[1], yc[1], zc[1], dx, dy, dz, ns, coords...)
    @parallel init_ϕi!(ϕ, ϕx, ϕy, ϕz)
    len_g   = sum_g(ϕ.==fluid)
    # visu
    if do_save
        (me==0) && !ispath("../out_visu") && mkdir("../out_visu")
        Vn  = @zeros(nx-2,ny-2,nz-2)
        τII = @zeros(nx-2,ny-2,nz-2)
        Ptv = @zeros(nx-2,ny-2,nz-2)
    end
    (me==0) && println("... done. Starting the real stuff 😎")
    # iteration loop
    err_V=2*ε_V; err_∇V=2*ε_∇V; iter=0; err_evo1=[]; err_evo2=[]
    while !((err_V <= ε_V) && (err_∇V <= ε_∇V)) && (iter <= maxiter)
        @parallel compute_P_τ!(∇V, Pt, τxx, τyy, τzz, τxy, τxz, τyz, Vx, Vy, Vz, ϕ, r, μ_veτ, Gdτ, dx, dy, dz)
        @hide_communication b_width begin
            @parallel compute_V!(Vx, Vy, Vz, Pt, τxx, τyy, τzz, τxy, τxz, τyz, ϕ, ρgx, ρgy, ρgz, dτ_ρ, dx, dy, dz)
            update_halo!(Vx,Vy,Vz)
        end
        iter += 1
        if iter % nchk == 0
            @parallel compute_Res!(Rx, Ry, Rz, Pt, τxx, τyy, τzz, τxy, τxz, τyz, ϕ, ρgx, ρgy, ρgz, dx, dy, dz)
            norm_Rx = norm_g(Rx)/psc*lz/sqrt(len_g)
            norm_Ry = norm_g(Ry)/psc*lz/sqrt(len_g)
            norm_Rz = norm_g(Rz)/psc*lz/sqrt(len_g)
            norm_∇V = norm_g(∇V)/vsc*lz/sqrt(len_g)
            err_V   = maximum([norm_Rx, norm_Ry, norm_Rz])
            err_∇V  = norm_∇V
            # push!(err_evo1, maximum([norm_Rx, norm_Ry, norm_∇V])); push!(err_evo2,iter/nx)
            if (me==0) @printf("# iters = %d, err_V = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_Rz=%1.3e], err_∇V = %1.3e \n", iter, err_V, norm_Rx, norm_Ry, norm_Rz, err_∇V) end
            GC.gc() # force garbage collection
        end
    end
    dim_g = (nx_g()-2, ny_g()-2, nz_g()-2)
    if do_save
        @parallel preprocess_visu!(Vn, τII, Ptv, Vx, Vy, Vz, τxx, τyy, τzz, τxy, τxz, τyz, Pt)
        @parallel apply_mask!(Vn, τII, Ptv, ϕ)
        out_path = "../out_visu"
        info     = MPI.Info()
        (me==0) && print("Saving HDF5 file...")
        h5open(joinpath(out_path, "results.h5"), "w", comm_cart, info) do io
            # Create dataset
            # Write local data
            ix = (coords[1]*(nx-2) + 1):(coords[1]+1)*(nx-2)
            iy = (coords[2]*(ny-2) + 1):(coords[2]+1)*(ny-2)
            iz = (coords[3]*(nz-2) + 1):(coords[3]+1)*(nz-2)
            ϕ_set = create_dataset(io, "/Phi", datatype(eltype(ϕ)), dataspace(dim_g))
            ϕ_set[ix,iy,iz] = Array(ϕ)[2:end-1,2:end-1,2:end-1]
            Vn_set = create_dataset(io, "/Vn", datatype(eltype(Vn)), dataspace(dim_g))
            Vn_set[ix,iy,iz] = Array(Vn)
            τII_set = create_dataset(io, "/TauII", datatype(eltype(τII)), dataspace(dim_g))
            τII_set[ix,iy,iz] = Array(τII)
            Pt_set = create_dataset(io, "/Pt", datatype(eltype(Ptv)), dataspace(dim_g))
            Pt_set[ix,iy,iz] = Array(Ptv)
        end
        (me==0) && println(" done")
        # write XDMF
        if me == 0
            xdoc = XMLDocument()
            xroot = create_root(xdoc, "Xdmf")
            set_attribute(xroot, "Version","3.0")

            xdomain = new_child(xroot, "Domain")
            xgrid   = new_child(xdomain, "Grid")
            set_attribute(xgrid, "GridType","Uniform")
            xtopo = new_child(xgrid, "Topology")
            set_attribute(xtopo, "TopologyType", "3DCoRectMesh")
            set_attribute(xtopo, "Dimensions", join(reverse(dim_g).+1,' '))

            xgeom = new_child(xgrid, "Geometry")
            set_attribute(xgeom, "GeometryType", "ORIGIN_DXDYDZ")

            xorig = new_child(xgeom, "DataItem")
            set_attribute(xorig, "Format", "XML")
            set_attribute(xorig, "NumberType", "Float")
            set_attribute(xorig, "Dimensions", "$(length(dim_g)) ")
            add_text(xorig, join((oz,oy,ox), ' '))

            xdr   = new_child(xgeom, "DataItem")
            set_attribute(xdr, "Format", "XML")
            set_attribute(xdr, "NumberType", "Float")
            set_attribute(xdr, "Dimensions", "$(length(dim_g))")
            add_text(xdr, join((dz,dy,dx), ' '))

            create_xdmf_attribute(xgrid,joinpath(out_path, "results.h5"),"Phi",dim_g)
            create_xdmf_attribute(xgrid,joinpath(out_path, "results.h5"),"Vn",dim_g)
            create_xdmf_attribute(xgrid,joinpath(out_path, "results.h5"),"TauII",dim_g)
            create_xdmf_attribute(xgrid,joinpath(out_path, "results.h5"),"Pt",dim_g)

            save_file(xdoc, joinpath(out_path, "results.xdmf3"))
        end
    end
    finalize_global_grid()
    return
end
# ---------------------
# preprocessing
# extract_geodata("Rhone"; do_rotate=true)

Stokes3D()
