const USE_GPU = haskey(ENV, "USE_GPU") ? parse(Bool, ENV["USE_GPU"]) : false
const gpu_id  = haskey(ENV, "GPU_ID" ) ? parse(Int , ENV["GPU_ID" ]) : 6
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
using Printf, Statistics, LinearAlgebra, MAT, Random, UnPack, Plots

include(joinpath(@__DIR__, "helpers3D_v2.jl"))

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
macro fm_xi(A) esc(:( 0.5*((($A[$ix,$iyi,$izi] != air)) + (($A[$ix+1,$iyi,$izi] != air))) )) end
macro fm_yi(A) esc(:( 0.5*((($A[$ixi,$iy,$izi] != air)) + (($A[$ixi,$iy+1,$izi] != air))) )) end
macro fm_zi(A) esc(:( 0.5*((($A[$ixi,$iyi,$iz] != air)) + (($A[$ixi,$iyi,$iz+1] != air))) )) end

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

@views function Stokes3D(inputs::InputParams3D)
    @unpack ϕ, x3rot, y3rot, z3rot, x3, y3, z3, xc, yc, zc, R, lx, ly, lz, nx, ny, nz, sc = inputs
    println("Ice flow solver: lx=$(round(lx,sigdigits=4)), ly=$(round(ly,sigdigits=4)), lz=$(round(lz,sigdigits=4)), sc=$(round(sc,sigdigits=4)))")
    # physics
    ## dimensionally independent
    # lz        = 1.0               # domain height    [m]
    μs0       = 1.0               # matrix viscosity [Pa*s]
    ρg0       = 1.0               # gravity          [Pa/m]
    ## scales
    psc       = ρg0*lz
    tsc       = μs0/psc
    vsc       = lz/tsc
    ## dimensionally dependent
    # lx        = lx_lz*lz
    # ly        = ly_lz*lz
    ρgv       = ρg0*R'*[0,0,1]
    ρgx,ρgy,ρgz = ρgv[1], ρgv[2], ρgv[3]
    # numerics
    maxiter   = 50nz         # maximum number of pseudo-transient iterations
    nchk      = 2*nz         # error checking frequency
    nviz      = 2*nz         # visualisation frequency
    ε_V       = 1e-6         # nonlinear absolute tolerance for momentum
    ε_∇V      = 1e-6         # nonlinear absolute tolerance for divergence
    CFL       = 0.95/sqrt(3) # stability condition
    Re        = 2π           # Reynolds number                     (numerical parameter #1)
    r         = 1.0          # Bulk to shear elastic modulus ratio (numerical parameter #2)
    # preprocessing
    dx,dy,dz  = lx/nx, ly/ny, lz/nz # cell sizes
    max_lxyz  = lz
    Vpdτ      = min(dx,dy,dz)*CFL
    dτ_ρ      = Vpdτ*max_lxyz/Re/μs0
    Gdτ       = Vpdτ^2/dτ_ρ/(r+2.0)
    μ_veτ     = 1.0/(1.0/Gdτ + 1.0/μs0)
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
    if do_save
        !ispath("../out_visu") && mkdir("../out_visu")
        matwrite("../out_visu/out_pa3D.mat", Dict("Phase"=> Array(ϕ), "x3rot"=> Array(x3rot), "y3rot"=> Array(y3rot), "z3rot"=> Array(z3rot), "xc"=> Array(xc), "yc"=> Array(yc), "zc"=> Array(zc), "rhogv"=> Array(ρgv), "lx"=> lx, "ly"=> ly, "lz"=> lz, "sc"=> sc); compress = true)
    end
    fntsz = 7; sl = ceil(Int,ny*0.2); xci, yci, zci = xc[2:end-1], yc[2:end-1], zc[2:end-1]
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
            norm_Rx = norm(Rx)/psc*lz/sqrt(length(Rx))
            norm_Ry = norm(Ry)/psc*lz/sqrt(length(Ry))
            norm_Rz = norm(Rz)/psc*lz/sqrt(length(Rz))
            norm_∇V = norm((ϕ.==fluid).*∇V)/vsc*lz/sqrt(length(∇V))
            err_V   = maximum([norm_Rx, norm_Ry, norm_Rz])
            err_∇V  = norm_∇V
            push!(err_evo1, maximum([norm_Rx, norm_Ry, norm_∇V])); push!(err_evo2,iter/nx)
            @printf("# iters = %d, err_V = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_Rz=%1.3e], err_∇V = %1.3e \n", iter, err_V, norm_Rx, norm_Ry, norm_Rz, err_∇V)
        end
        if do_visu && iter % nviz == 0
            @parallel preprocess_visu!(Vn, τII, Vx, Vy, Vz, τxx, τyy, τzz, τxy, τxz, τyz)
            Vn_s  .=  Vn[:,sl,:];  Vn_s[Vn_s .==0] .= NaN
            τII_s .= τII[:,sl,:]; τII_s[τII_s.==0] .= NaN
            Pt_s  .=  Pt[:,sl,:];  Pt_s[Pt_s .==0] .= NaN
            p1 = heatmap(xc ,zc ,Array(Vn_s)' ; c=:batlow, title="Vn (y=0)", opts...)
            p2 = heatmap(xci,zci,Array(τII_s)'; c=:batlow, title="τII (y=0)", opts...)
            p3 = heatmap(xc, zc ,Array(Pt_s)' ; c=:viridis,title="Pressure (y=0)", opts...)
            p4 = plot(err_evo2,err_evo1; legend=false, xlabel="# iterations/nx", ylabel="log10(error)", labels="max(error)", opts2...)
            display(plot(p1, p2, p3, p4, size=(1e3,600), dpi=200,layout=(2,2)))
        end
    end
    if do_save
        @parallel preprocess_visu!(Vn, τII, Vx, Vy, Vz, τxx, τyy, τzz, τxy, τxz, τyz)
        Vn_v  .= Vn;  Vn_v[Vn_v.==0]   .= NaN
        τII_v .= τII; τII_v[τII_v.==0] .= NaN
        Pt_v  .= Pt;  Pt_v[Pt_v.==0]   .= NaN
        matwrite("../out_visu/out_res3D.mat", Dict("Vn"=> Array(Vn), "tII"=> Array(τII), "Pt"=> Array(Pt), "xc"=> Array(xc), "yc"=> Array(yc), "zc"=> Array(zc)); compress = true)
    end
    return
end

# ---------------------

# preprocessing
inputs = preprocess("../data/arolla3D.mat"; resx=128, resy=128, fact_nz=2)

@time Stokes3D(inputs)
