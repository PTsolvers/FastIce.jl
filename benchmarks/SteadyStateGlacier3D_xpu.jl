const USE_GPU = haskey(ENV, "USE_GPU") ? parse(Bool, ENV["USE_GPU"]) : false
const gpu_id  = haskey(ENV, "GPU_ID" ) ? parse(Int , ENV["GPU_ID" ]) : 7
const do_save = haskey(ENV, "DO_SAVE") ? parse(Bool, ENV["DO_SAVE"]) : true
const do_visu = haskey(ENV, "DO_VISU") ? parse(Bool, ENV["DO_VISU"]) : true
###
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
    CUDA.device!(gpu_id)
else
    @init_parallel_stencil(Threads, Float64, 3)
end
using ImplicitGlobalGrid, Printf, Statistics, LinearAlgebra, Random, UnPack, Plots, MAT, WriteVTK
import MPI

include(joinpath(@__DIR__, "helpers3D_v4_xpu.jl"))

import ParallelStencil: INDICES
ix,iy,iz    = INDICES[1], INDICES[2], INDICES[3]
ixi,iyi,izi = :($ix+1), :($iy+1), :($iz+1)

const air   = 0.0
const fluid = 1.0
const solid = 2.0

macro fm(A)   esc(:( $A[$ix,$iy,$iz] == fluid )) end
macro fmxy(A) esc(:( !($A[$ix,$iy,$izi] == air || $A[$ix+1,$iy,$izi] == air || $A[$ix,$iy+1,$izi] == air || $A[$ix+1,$iy+1,$izi] == air) )) end
macro fmxz(A) esc(:( !($A[$ix,$iyi,$iz] == air || $A[$ix+1,$iyi,$iz] == air || $A[$ix,$iyi,$iz+1] == air || $A[$ix+1,$iyi,$iz+1] == air) )) end
macro fmyz(A) esc(:( !($A[$ixi,$iy,$iz] == air || $A[$ixi,$iy+1,$iz] == air || $A[$ixi,$iy,$iz+1] == air || $A[$ixi,$iy+1,$iz+1] == air) )) end

@parallel function compute_P_τ!(∇V, Pt, τxx, τyy, τzz, τxy, τxz, τyz, Vx, Vy, Vz, ϕ, r, μ_veτ, Gdτ, dx, dy, dz)
    @all(∇V)  = @d_xa(Vx)/dx + @d_ya(Vy)/dy + @d_za(Vz)/dz
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

@parallel function preprocess_visu!(Vn, τII, Vx, Vy, Vz, τxx, τyy, τzz, τxy, τxz, τyz)
    @all(Vn)  = (@av_xa(Vx)*@av_xa(Vx) + @av_ya(Vy)*@av_ya(Vy) + @av_za(Vz)*@av_za(Vz))^0.5
    @all(τII) = (0.5*(@inn(τxx)*@inn(τxx) + @inn(τyy)*@inn(τyy) + @inn(τzz)*@inn(τzz)) + @av_xya(τxy)*@av_xya(τxy) + @av_xza(τxz)*@av_xza(τxz) + @av_yza(τyz)*@av_yza(τyz))^0.5
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

@views function Stokes3D()
    # inputs
    filename  = "../data/alps/data_Rhone.h5"
    resx      = 64
    resy      = 64
    resz      = 32
    do_nondim = true
    fact_nz   = 2
    ns        = 4
    olen      = 1
    dim       = (1,1,1)

    nx = gpu_res(resx , tx, olen)
    ny = gpu_res(resy , ty, olen)
    nz = gpu_res(resz , tz, olen)

    me, dims, nprocs, coords = init_global_grid(nx, ny, nz; dimx=dim[1], dimy=dim[2], dimz=dim[3]) # MPI initialisation

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
    sc         = do_nondim ? 1.0/lz : 1.0
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
    maxiter   = 50nz         # maximum number of pseudo-transient iterations
    nchk      = 2*nz         # error checking frequency
    nviz      = 2*nz         # visualisation frequency
    ε_V       = 1e-8         # nonlinear absolute tolerance for momentum
    ε_∇V      = 1e-8         # nonlinear absolute tolerance for divergence
    CFL       = 0.95/sqrt(3) # stability condition
    Re        = 2π           # Reynolds number                     (numerical parameter #1)
    r         = 1.0          # Bulk to shear elastic modulus ratio (numerical parameter #2)
    
    # preprocessing
    xv_d, yv_d = x2v[:,1], y2v[1,:]
    xv, yv = LinRange(xv_d[1], xv_d[end], ns*(nx_g()+1)), LinRange(yv_d[1], yv_d[end], ns*(ny_g()+1))

    zbed2, zthick2 = interp(zbed, zthick, xv_d, yv_d, xv, yv)
    println("- interpolate original data (nxv, nyv = $(size(zbed)[1]), $(size(zbed)[2])) on nxv, nyv = $(size(zbed2)[1]), $(size(zbed2)[2]) grid ($(ns)x oversampling)")
    
    nsmb, nsmt = 5, 5 #ceil(Int,nx/20)
    println("- apply smoothing ($nsmb steps on bed, $nsmt steps on thickness)")
    Tmp = copy(zbed2);   for ismb=1:nsmb smooth2D!(zbed2  , Tmp, 1.0)  end
    Tmp = copy(zthick2); for ismt=1:nsmt smooth2D!(zthick2, Tmp, 1.0)  end

    # reconstruct surface
    zsurf2 = zbed2 .+ zthick2

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
    Vn        = @zeros(nx  ,ny  ,nz  )
    τII       = @zeros(nx-2,ny-2,nz-2)
    # visu
    Vn_v      = @zeros(nx,ny,nz) # visu
    τII_v     = copy(τII) # visu
    Pt_v      = copy(Pt)  # visu
    Vn_s      = @zeros(nx  ,nz  ) # visu
    τII_s     = @zeros(nx-2,nz-2) # visu
    Pt_s      = @zeros(nx  ,nz  ) # visu
    Rx1_v     = zeros(nx-1,nz-2) # visu
    Ry1_v     = zeros(nx-2,nz-2) # visu
    Rz1_v     = zeros(nx-2,nz-1) # visu
    Rx2_v     = zeros(ny-2,nz-2) # visu
    Ry2_v     = zeros(ny-1,nz-2) # visu
    Rz2_v     = zeros(ny-2,nz-1) # visu


    # rotate grid
    coords   = Data.Array(coords)
    Rinv     = Data.Array(R')
    xc_g     = Data.Array(xc)
    yc_g     = Data.Array(yc)
    zc_g     = Data.Array(zc)
    zsurf_g  = Data.Array(zsurf2)
    zbed_g   = Data.Array(zbed2)

    X3rot_g  = @zeros(nx,ny,nz)
    Y3rot_g  = @zeros(nx,ny,nz)
    Z3rot_g  = @zeros(nx,ny,nz)
    
    @parallel my_rot_d!(X3rot_g, Y3rot_g, Z3rot_g, Rinv, xc_g, yc_g, zc_g, coords)

    # set phases
    println("- set phases (0-air, 1-ice, 2-bedrock)")
    ϕ      = air .* @ones(size(x3))
    @parallel set_phases!(ϕ, X3rot, Y3rot, Z3rot, zsurf_g, zbed_g, -xrmax, -yrmax, dx, dy, ns)


    @parallel init_ϕi!(ϕ,ϕx,ϕy,ϕz)
    if do_save
        !ispath("../out_visu") && mkdir("../out_visu")
        # matwrite("../out_visu/out_pa3D.mat", Dict("Phase"=> Array(ϕ), "x3rot"=> Array(x3rot), "y3rot"=> Array(y3rot), "z3rot"=> Array(z3rot), "xc"=> Array(xc), "yc"=> Array(yc), "zc"=> Array(zc), "rhogv"=> Array(ρgv), "lx"=> lx, "ly"=> ly, "lz"=> lz, "sc"=> sc); compress = true)
    end
    fntsz = 16; sl = ceil(Int,ny*0.2); xci, yci, zci = xc[2:end-1], yc[2:end-1], zc[2:end-1]
    xvi, yvi, zvi = 0.5.*(xc[1:end-1] .+ xc[2:end]), 0.5.*(yc[1:end-1] .+ yc[2:end]), 0.5.*(zc[1:end-1] .+ zc[2:end])
    opts  = (aspect_ratio=1, xlims=(xc[1],xc[end]), ylims=(zc[1],zc[end]), yaxis=font(fntsz,"Courier"), xaxis=font(fntsz,"Courier"), framestyle=:box, titlefontsize=fntsz, titlefont="Courier")
    opts2 = (linewidth=2, markershape=:circle, markersize=3,yaxis = (:log10, font(fntsz,"Courier")), xaxis=font(fntsz,"Courier"), framestyle=:box, titlefontsize=fntsz, titlefont="Courier")
    # iteration loop
    err_V=2*ε_V; err_∇V=2*ε_∇V; iter=0; err_evo1=[]; err_evo2=[]
    while !((err_V <= ε_V) && (err_∇V <= ε_∇V)) && (iter <= maxiter)
        @parallel compute_P_τ!(∇V, Pt, τxx, τyy, τzz, τxy, τxz, τyz, Vx, Vy, Vz, ϕ, r, μ_veτ, Gdτ, dx, dy, dz)
        @parallel compute_V!(Vx, Vy, Vz, Pt, τxx, τyy, τzz, τxy, τxz, τyz, ϕ, ρgx, ρgy, ρgz, dτ_ρ, dx, dy,dz)
        iter += 1
        if iter % nchk == 0
            @parallel compute_Res!(Rx, Ry, Rz, Pt, τxx, τyy, τzz, τxy, τxz, τyz, ϕ, ρgx, ρgy, ρgz, dx, dy, dz)
            norm_Rx = norm((ϕx.==fluid).*Rx)/psc*lz/sqrt(length(Rx))
            norm_Ry = norm((ϕy.==fluid).*Ry)/psc*lz/sqrt(length(Ry))
            norm_Rz = norm((ϕz.==fluid).*Rz)/psc*lz/sqrt(length(Rz))
            norm_∇V = norm((ϕ.==fluid).*∇V)/vsc*lz/sqrt(length(∇V))
            err_V   = maximum([norm_Rx, norm_Ry, norm_Rz])
            err_∇V  = norm_∇V
            push!(err_evo1, maximum([norm_Rx, norm_Ry, norm_∇V])); push!(err_evo2,iter/nx)
            @printf("# iters = %d, err_V = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_Rz=%1.3e], err_∇V = %1.3e \n", iter, err_V, norm_Rx, norm_Ry, norm_Rz, err_∇V)
        end
        if do_visu && iter % nviz == 0
            @parallel preprocess_visu!(Vn, τII, Vx, Vy, Vz, τxx, τyy, τzz, τxy, τxz, τyz)
            Vn_s  .=  Vn[:,sl,:];  Vn_s[ϕ[:,sl,:].!=fluid] .= NaN
            τII_s .= τII[:,sl,:]; τII_s[τII_s.==0] .= NaN
            Pt_s  .=  Pt[:,sl,:];  Pt_s[ϕ[:,sl,:].!=fluid] .= NaN
            Rx1_v .= Rx[:,sl,:]; Rx1_v[ϕx[:,sl,:].!=fluid] .= NaN
            Ry1_v .= Ry[:,sl,:]; Ry1_v[ϕy[:,sl,:].!=fluid] .= NaN
            Rz1_v .= Rz[:,sl,:]; Rz1_v[ϕz[:,sl,:].!=fluid] .= NaN
            Rx2_v .= Rx[sl,:,:]; Rx2_v[ϕx[sl,:,:].!=fluid] .= NaN
            Ry2_v .= Ry[sl,:,:]; Ry2_v[ϕy[sl,:,:].!=fluid] .= NaN
            Rz2_v .= Rz[sl,:,:]; Rz2_v[ϕz[sl,:,:].!=fluid] .= NaN
            p1 = heatmap(xvi,zci,Rx1_v'; c=:batlow, title="Rx (y=0)", opts...)
            p2 = heatmap(xci,zci,Ry1_v'; c=:batlow, title="Ry (y=0)", opts...)
            p3 = heatmap(xci,zvi,Rz1_v'; c=:batlow, title="Rz (y=0)", opts...)
            p4 = heatmap(yci,zci,Rx2_v'; c=:batlow, title="Rx (x=0)", opts...)
            p5 = heatmap(yvi,zci,Ry2_v'; c=:batlow, title="Ry (x=0)", opts...)
            p6 = heatmap(yci,zvi,Rz2_v'; c=:batlow, title="Rz (x=0)", opts...)
            p7 = heatmap(xc ,zc ,Array(Vn_s)' ; c=:batlow, title="Vn (y=0)", opts...)
            # p2 = heatmap(xci,zci,Array(τII_s)'; c=:batlow, title="τII (y=0)", opts...)
            p8 = heatmap(xc, zc ,Array(Pt_s)' ; c=:viridis,title="Pressure (y=0)", opts...)
            p9 = plot(err_evo2,err_evo1; legend=false, xlabel="# iterations/nx", ylabel="log10(error)", labels="max(error)", opts2...)
            display(plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, size=(1.5e3,8e2), dpi=200))
        end
    end
    if do_save
        @parallel preprocess_visu!(Vn, τII, Vx, Vy, Vz, τxx, τyy, τzz, τxy, τxz, τyz)
        Vn_v  .= Vn;  Vn_v[Vn_v.==0]   .= NaN
        τII_v .= τII; τII_v[τII_v.==0] .= NaN
        Pt_v  .= Pt;  Pt_v[Pt_v.==0]   .= NaN
        # matwrite("../out_visu/out_res3D.mat", Dict("Vn"=> Array(Vn), "tII"=> Array(τII), "Pt"=> Array(Pt), "xc"=> Array(xc), "yc"=> Array(yc), "zc"=> Array(zc)); compress = true)
        st = 1 # downsampling factor
        vtk_grid("../out_visu/out_3Dfields", Array(x3rot)[1:st:end,1:st:end,1:st:end], Array(y3rot)[1:st:end,1:st:end,1:st:end], Array(z3rot)[1:st:end,1:st:end,1:st:end]; compress=5) do vtk
            vtk["Vnorm"]    = Array(Vn_v)[1:st:end,1:st:end,1:st:end]
            vtk["TauII"]    = Array(τII_v)[1:st:end,1:st:end,1:st:end]
            vtk["Pressure"] = Array(Pt_v)[1:st:end,1:st:end,1:st:end]
            vtk["Phase"]    = Array(ϕ)[1:st:end,1:st:end,1:st:end]
        end
    end
    return
end

# ---------------------

# preprocessing
# extract_geodata("Rhone"; do_rotate=true)

@time Stokes3D()
