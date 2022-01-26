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
using Plots, Printf, Statistics, LinearAlgebra, MAT, Random

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

macro sm_xi(A) esc(:( ($A[$ix,$iyi,$izi] == fluid) && ($A[$ix+1,$iyi,$izi] == fluid) )) end
macro sm_yi(A) esc(:( ($A[$ixi,$iy,$izi] == fluid) && ($A[$ixi,$iy+1,$izi] == fluid) )) end
macro sm_zi(A) esc(:( ($A[$ixi,$iyi,$iz] == fluid) && ($A[$ixi,$iyi,$iz+1] == fluid) )) end

macro fm_xi(A) esc(:( 0.5*((($A[$ix,$iyi,$izi] != air)) + (($A[$ix+1,$iyi,$izi] != air))) )) end
macro fm_yi(A) esc(:( 0.5*((($A[$ixi,$iy,$izi] != air)) + (($A[$ixi,$iy+1,$izi] != air))) )) end
macro fm_zi(A) esc(:( 0.5*((($A[$ixi,$iyi,$iz] != air)) + (($A[$ixi,$iyi,$iz+1] != air))) )) end

@parallel function compute_V!(Vx, Vy, Vz, Pt, τxx, τyy, τzz, τxy, τxz, τyz, ϕ, ρgx, ρgy, ρgz, dτ_ρ, dx, dy, dz)
    @inn(Vx) = @sm_xi(ϕ)*( @inn(Vx) + dτ_ρ*(@d_xi(τxx)/dx + @d_ya(τxy)/dy + @d_za(τxz)/dz - @d_xi(Pt)/dx - ρgx) )
    @inn(Vy) = @sm_yi(ϕ)*( @inn(Vy) + dτ_ρ*(@d_yi(τyy)/dy + @d_xa(τxy)/dx + @d_za(τyz)/dz - @d_yi(Pt)/dy - ρgy) )
    @inn(Vz) = @sm_zi(ϕ)*( @inn(Vz) + dτ_ρ*(@d_zi(τzz)/dy + @d_xa(τxz)/dx + @d_ya(τyz)/dy - @d_zi(Pt)/dz - ρgz) )
    return
end

@parallel function compute_Res!(Rx, Ry, Rz, Pt, τxx, τyy, τzz, τxy, τxz, τyz, ϕ, ρgx, ρgy, ρgz, dx, dy, dz)
    @all(Rx)  = @sm_xi(ϕ)*(@d_xi(τxx)/dx + @d_ya(τxy)/dy + @d_za(τxz)/dz - @d_xi(Pt)/dx - @fm_xi(ϕ)*ρgx)
    @all(Ry)  = @sm_yi(ϕ)*(@d_yi(τyy)/dy + @d_xa(τxy)/dx + @d_za(τyz)/dz - @d_yi(Pt)/dy - @fm_yi(ϕ)*ρgy)
    @all(Rz)  = @sm_zi(ϕ)*(@d_zi(τzz)/dy + @d_xa(τxz)/dx + @d_ya(τyz)/dy - @d_zi(Pt)/dz - @fm_zi(ϕ)*ρgz)
    return
end

function is_inside_fluid(x,y,z,gl)
    return (x+0.1*gl)*(x+0.1*gl) + z*z < gl*gl
end

function is_inside_solid(x,y,z,lx,ly,amp,ω,tanβ,el)
    return z < amp*sin(ω*x/lx)*sin(ω*y/ly) + tanβ*x + el + y^2/ly
end

@parallel_indices (ix,iy,iz) function init_ϕ!(ϕ,gl,el,tanβ,ω,amp,dx,dy,dz,lx,ly,lz)
    xc,yc,zc = dx*ix-dx/2-lx/2, dy*iy-dy/2-ly/2, dz*iz-dz/2
    if checkbounds(Bool,ϕ,ix,iy,iz)
        ϕ[ix,iy,iz] = air
        if is_inside_fluid(xc,yc,zc,gl)
            ϕ[ix,iy,iz] = fluid
        end
        if is_inside_solid(xc,yc,zc,lx,ly,amp,ω,tanβ,el)
            ϕ[ix,iy,iz] = solid
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

@views function Stokes3D()
    # physics
    ## dimensionally independent
    lz        = 1.0               # domain height    [m]
    μs0       = 1.0               # matrix viscosity [Pa*s]
    ρg0       = 1.0               # gravity          [Pa/m]
    ## scales
    psc       = ρg0*lz
    tsc       = μs0/psc
    vsc       = lz/tsc
    ## nondimensional parameters
    lx_lz     = 2.0
    ly_lz     = 2.0
    gl_lz     = 0.9
    el_lz     = 0.05
    amp_lz    = 1/25
    α         = 0*π/12
    tanβ      = tan(-π/12)
    ωlz       = 10π
    ## dimensionally dependent
    lx        = lx_lz*lz
    ly        = ly_lz*lz
    gl        = gl_lz*lz
    el        = el_lz*lz
    amp       = amp_lz*lz
    ρgx       = ρg0*sin(α)
    ρgy       = ρg0*sin(α)
    ρgz       = ρg0*cos(α)
    ω         = ωlz/lz
    # numerics
    nz        = 64
    nx        = ceil(Int,lx_lz*nz)
    ny        = ceil(Int,ly_lz*nz)
    nx,ny,nz  = nx-1,ny-1,nz-1
    maxiter   = 50nz         # maximum number of pseudo-transient iterations
    nchk      = 2*nz         # error checking frequency
    nviz      = 2*nz         # visualisation frequency
    ε_V       = 1e-3         # nonlinear absolute tolerance for momentum
    ε_∇V      = 1e-3         # nonlinear absolute tolerance for divergence
    CFL       = 0.95/sqrt(3) # stability condition
    Re        = 2π           # Reynolds number                     (numerical parameter #1)
    r         = 1.0          # Bulk to shear elastic modulus ratio (numerical parameter #2)
    # preprocessing
    dx,dy,dz  = lx/nx, ly/ny, lz/nz # cell sizes
    max_lxyz  = 0.5gl
    Vpdτ      = min(dx,dy,dz)*CFL
    dτ_ρ      = Vpdτ*max_lxyz/Re/μs0
    Gdτ       = Vpdτ^2/dτ_ρ/(r+2.0)
    μ_veτ     = 1.0/(1.0/Gdτ + 1.0/μs0)
    Xc,Yc,Zc  = LinRange(-(lx-dx)/2,(lx-dx)/2,nx  ), LinRange(-(ly-dy)/2,(ly-dy)/2,ny  ), LinRange(dz/2,lz-dz/2,nz  )
    Xv,Yv,Zv  = LinRange(- lx/2    , lx/2    ,nx+1), LinRange(- ly/2    , ly/2    ,ny+1), LinRange(0   ,lz     ,nz+1)
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
    ϕ         = @zeros(nx  ,ny  ,nz  )
    Vx1_v     = zeros(nx+1,nz  ) # visu
    Vy1_v     = zeros(nx  ,nz  ) # visu
    Vz1_v     = zeros(nx  ,nz+1) # visu
    Pt1_v     = zeros(nx  ,nz  ) # visu
    Vx2_v     = zeros(ny  ,nz  ) # visu
    Vy2_v     = zeros(ny+1,nz  ) # visu
    Vz2_v     = zeros(ny  ,nz+1) # visu
    Pt2_v     = zeros(ny  ,nz  ) # visu
    @parallel init_ϕ!(ϕ,gl,el,tanβ,ω,amp,dx,dy,dz,lx,ly,lz)
    @parallel init_ϕi!(ϕ,ϕx,ϕy,ϕz)
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
            push!(err_evo1, maximum([norm_Rx, norm_Ry, norm_∇V])); push!(err_evo2,iter/ny)
            @printf("# iters = %d, err_V = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_Rz=%1.3e], err_∇V = %1.3e \n", iter, err_V, norm_Rx, norm_Ry, norm_Rz, err_∇V)
        end
        if iter % nviz == 0
            iy_sl = 16
            Vx1_v .= Vx[:,iy_sl,:]; Vx1_v[Vx1_v.==0] .= NaN
            Vy1_v .= Vy[:,iy_sl,:]; Vy1_v[Vy1_v.==0] .= NaN
            Vz1_v .= Vz[:,iy_sl,:]; Vz1_v[Vz1_v.==0] .= NaN
            Pt1_v .= Pt[:,iy_sl,:]; Pt1_v[Pt1_v.==0] .= NaN
            Vx2_v .= Vx[iy_sl,:,:]; Vx2_v[Vx2_v.==0] .= NaN
            Vy2_v .= Vy[iy_sl,:,:]; Vy2_v[Vy2_v.==0] .= NaN
            Vz2_v .= Vz[iy_sl,:,:]; Vz2_v[Vz2_v.==0] .= NaN
            Pt2_v .= Pt[iy_sl,:,:]; Pt2_v[Pt2_v.==0] .= NaN
            fntsz = 12
            opts  = (aspect_ratio=1, xlims=(Xv[1],Xv[end]), ylims=(Zc[1],Zc[end]), yaxis=font(fntsz,"Courier"), xaxis=font(fntsz,"Courier"), framestyle=:box, titlefontsize=fntsz, titlefont="Courier")
            opts2 = (linewidth=2, markershape=:circle, markersize=3,yaxis = (:log10, font(fntsz,"Courier")), xaxis=font(fntsz,"Courier"), framestyle=:box, titlefontsize=fntsz, titlefont="Courier")
            p1 = heatmap(Xv,Zc,Vx1_v'; c=:batlow, title="Vx (y=0)", opts...)
            p2 = heatmap(Xc,Zc,Vy1_v'; c=:batlow, title="Vy (y=0)", opts...)
            p3 = heatmap(Xc,Zv,Vz1_v'; c=:batlow, title="Vz (y=0)", opts...)
            p4 = heatmap(Yc,Zc,Vx2_v'; c=:batlow, title="Vx (x=0)", opts...)
            p5 = heatmap(Yv,Zc,Vy2_v'; c=:batlow, title="Vy (x=0)", opts...)
            p6 = heatmap(Yc,Zv,Vz2_v'; c=:batlow, title="Vz (x=0)", opts...)
            p7 = heatmap(Xc,Zc,Pt1_v'; c=:viridis, title="Pressure (y=0)", opts...)
            p8 = heatmap(Yc,Zc,Pt2_v'; c=:viridis, title="Pressure (x=0)", opts...)
            p9 = plot(err_evo2,err_evo1; legend=false, xlabel="# iterations/nx", ylabel="log10(error)", labels="max(error)", opts2...)
            display(plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, size=(3e3,1.5e3), dpi=200,layout=(3,3)))
        end
    end
    return
end

Stokes3D()
