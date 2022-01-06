const USE_GPU = haskey(ENV, "USE_GPU") ? parse(Bool, ENV["USE_GPU"]) : false
const gpu_id  = haskey(ENV, "GPU_ID" ) ? parse(Int , ENV["GPU_ID" ]) : 0
###
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDA.device!(gpu_id)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, Statistics, LinearAlgebra, MAT, Random

import ParallelStencil: INDICES
ix,iy = INDICES[1], INDICES[2]
ixi,iyi = :($ix+1), :($iy+1)

const air   = 0.0
const fluid = 1.0
const solid = 2.0

macro fm(A)      esc(:( $A[$ix,$iy] == fluid )) end
macro fa(A)      esc(:( $A[$ix,$iy] == air )) end
macro fmxy_xi(A) esc(:( !(($A[$ix,$iy] == air && $A[$ix,$iy+1] == air) || ($A[$ix+1,$iy] == air && $A[$ix+1,$iy+1] == air)) )) end
macro fmxy_yi(A) esc(:( !(($A[$ix,$iy] == air && $A[$ix+1,$iy] == air) || ($A[$ix,$iy+1] == air && $A[$ix+1,$iy+1] == air)) )) end

macro εxx() esc(:( @d_xa(Vx)/dx )) end
macro εyy() esc(:( @d_ya(Vy)/dy )) end
macro εxy() esc(:( 0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx) )) end

macro fnai(A)      esc(:( $A[$ixi,$iyi] != air                 )) end
macro Gdτ()        esc(:(Vpdτ*Re_mech*@all(μs)/max_lxy/(r+2.0) )) end
macro Gdτ_av()     esc(:(Vpdτ*Re_mech*@av(μs)/max_lxy/(r+2.0)  )) end
macro μ_veτ()      esc(:(1.0/(1.0/@Gdτ()    + 1.0/@all(μs))    )) end
macro μ_veτ_av()   esc(:(1.0/(1.0/@Gdτ_av() + 1.0/@av(μs))     )) end

@parallel_indices (ix,iy) function compute_EII!(EII, Vx, Vy, ϕ, dx, dy)
    nfluid = 0
    eii    = 0.0
    if ix <= size(EII,1)-2 && iy <= size(EII,2)-2
        if ϕ[ix,iy] == fluid && ϕ[ix+1,iy] == fluid && ϕ[ix,iy+1] == fluid && ϕ[ix+1,iy+1] == fluid
            nfluid += 1
            eii    += (Vx[ix+1,iy+1] - Vx[ix+1,iy])/dy + (Vy[ix+1,iy+1] - Vy[ix,iy+1])/dx
        end
        if ϕ[ix+1,iy] == fluid && ϕ[ix+2,iy] == fluid && ϕ[ix+1,iy+1] == fluid && ϕ[ix+2,iy+1] == fluid
            nfluid += 1
            eii    += (Vx[ix+2,iy+1] - Vx[ix+2,iy])/dy + (Vy[ix+2,iy+1] - Vy[ix+1,iy+1])/dx
        end
        if ϕ[ix,iy+1] == fluid && ϕ[ix+1,iy+1] == fluid && ϕ[ix,iy+2] == fluid && ϕ[ix+1,iy+2] == fluid
            nfluid += 1
            eii    += (Vx[ix+1,iy+2] - Vx[ix+1,iy+1])/dy + (Vy[ix+1,iy+2] - Vy[ix,iy+2])/dx
        end
        if ϕ[ix+1,iy+1] == fluid && ϕ[ix+2,iy+1] == fluid && ϕ[ix+1,iy+2] == fluid && ϕ[ix+2,iy+2] == fluid
            nfluid += 1
            eii    += (Vx[ix+2,iy+2] - Vx[ix+2,iy+1])/dy + (Vy[ix+2,iy+2] - Vy[ix+1,iy+2])/dx
        end
        if nfluid > 0
            eii /= nfluid
        end
        EII[ix+1,iy+1] = eii
    end
    return
end

@parallel function compute_P_τ_qT!(∇V, Pt, τxx, τyy, τxy, Vx, Vy, μs, qTx, qTy, T, ϕ, Vpdτ, Re_mech, r, max_lxy, χ, θr_dτ, dx, dy)
    # mechanics
    @all(∇V)  = @d_xa(Vx)/dx + @d_ya(Vy)/dy
    @all(Pt)  = @fm(ϕ)*(@all(Pt) - r*@Gdτ()*@all(∇V))
    @all(τxx) =                  @fm(ϕ)*2.0*@μ_veτ()*(@εxx() + @all(τxx)/@Gdτ()/2.0)
    @all(τyy) =                  @fm(ϕ)*2.0*@μ_veτ()*(@εyy() + @all(τyy)/@Gdτ()/2.0)
    @all(τxy) = @fmxy_xi(ϕ)*@fmxy_yi(ϕ)*2.0*@μ_veτ_av()*(@εxy() + @all(τxy)/@Gdτ_av()/2.0)
    # thermo
    @inn_x(qTx) = (@inn_x(qTx) * θr_dτ - χ*@d_xa(T)/dx) / (θr_dτ + 1.0)
    @inn_y(qTy) = (@inn_y(qTy) * θr_dτ - χ*@d_ya(T)/dy) / (θr_dτ + 1.0)
    return
end

macro sm_xi(A) esc(:( !(($A[$ix,$iyi] == solid) || ($A[$ix+1,$iyi] == solid)) )) end
macro sm_yi(A) esc(:( !(($A[$ixi,$iy] == solid) || ($A[$ixi,$iy+1] == solid)) )) end
macro fm_xi(A) esc(:( 0.5*((($A[$ix,$iyi] != air)) + (($A[$ix+1,$iyi] != air))) )) end
macro fm_yi(A) esc(:( 0.5*((($A[$ixi,$iy] != air)) + (($A[$ixi,$iy+1] != air))) )) end

macro dτ_ρ_mech_ax() esc(:( Vpdτ*max_lxy/Re_mech/@av_xi(μs) )) end
macro dτ_ρ_mech_ay() esc(:( Vpdτ*max_lxy/Re_mech/@av_yi(μs) )) end

@parallel function compute_V_T!(Vx, Vy, Pt, τxx, τyy, τxy, EII, μs, T, T_o, qTx, qTy, ϕ, ρgx, ρgy, μs0, T0, Ta, Q_R, dt, Vpdτ, max_lxy, Re_mech, dτ_ρ_heat, dx, dy)
    # mechanics
    @inn(Vx) = @sm_xi(ϕ)*( @inn(Vx) + @dτ_ρ_mech_ax()*(@d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(Pt)/dx - @fm_xi(ϕ)*ρgx) )
    @inn(Vy) = @sm_yi(ϕ)*( @inn(Vy) + @dτ_ρ_mech_ay()*(@d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(Pt)/dy - @fm_yi(ϕ)*ρgy) )
    # thermo
    @all(T)  = (@all(T) + dτ_ρ_heat *(@all(T_o)/dt - @d_xa(qTx)/dx - @d_ya(qTy)/dy + 2.0*@all(μs)*@all(EII)))/(1.0 + dτ_ρ_heat/dt)
    @all(T)  = @fa(ϕ)*Ta + (1.0 - @fa(ϕ))*@all(T)
    @all(μs) = μs0*exp(-Q_R*(1.0 - T0/@all(T)))
    return
end

@parallel function compute_Res!(Rx, Ry, RT, Pt, τxx, τyy, τxy, T, T_o, qTx, qTy, EII, μs, ϕ, ρgx, ρgy, dt, dx, dy)
    # mechanics
    @all(Rx) = @sm_xi(ϕ)*(@d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(Pt)/dx - @fm_xi(ϕ)*ρgx)
    @all(Ry) = @sm_yi(ϕ)*(@d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(Pt)/dy - @fm_yi(ϕ)*ρgy)
    # thermo
    @all(RT) = -(@all(T) - @all(T_o))/dt - (@d_xa(qTx)/dx + @d_ya(qTy)/dy) + 2.0*@all(μs)*@all(EII)
    return
end

function is_inside_fluid(x,y,gl)
    return (x+0.1*gl)*(x+0.1*gl) + y*y < gl*gl
end

function is_inside_solid(x,y,lx,amp,ω,tanβ,el)
    return y < amp*sin(ω*x/lx) + tanβ*x + el
end

@parallel_indices (ix,iy) function init_ϕ!(ϕ,gl,el,tanβ,ω,amp,dx,dy,lx,ly)
    xc,yc = dx*ix-dx/2-lx/2, dy*iy-dy/2
    if checkbounds(Bool,ϕ,ix,iy)
        ϕ[ix,iy] = air
        if is_inside_fluid(xc,yc,gl)
            ϕ[ix,iy] = fluid
        end
        if is_inside_solid(xc,yc,lx,amp,ω,tanβ,el)
            ϕ[ix,iy] = solid
        end
    end
    return
end

@parallel_indices (ix,iy) function init_T!(T,ϕ,T0,Ta)
    if checkbounds(Bool,T,ix,iy)
        T[ix,iy] = (ϕ[ix,iy] == air) ? Ta : T0
    end
    return
end


@views function Stokes2D()
    # physics
    ## dimensionally independent
    ly        = 1.0  # domain height                                     [m]
    μs0       = 1.0  # matrix viscosity                                  [Pa*s]
    ρg0       = 1.0  # gravity                                           [Pa/m]
    ΔT        = 1.0  # temperature difference between ice and atmosphere [K]
    ## scales
    psc       = ρg0*ly
    tsc       = μs0/psc
    vsc       = ly/tsc
    ## nondimensional parameters
    lx_ly     = 2.0
    gl_ly     = 0.9
    el_ly     = 0.3
    amp_ly    = 1/25
    α         = 0*π/12
    tanβ      = tan(-π/12)
    ωly       = 10π
    T0_δT     = 1.0
    Q_R       = 1.0e1
    ## dimensionally dependent
    lx        = lx_ly*ly
    gl        = gl_ly*ly
    el        = el_ly*ly
    amp       = amp_ly*ly
    ρgx       = ρg0*sin(α)
    ρgy       = ρg0*cos(α)
    ω         = ωly/ly
    χ         = 1e-4*ly^2/tsc # m^2/s = ly^3 * ρg0 / μs0
    T0        = T0_δT*ΔT
    dt        = 1e-2*tsc
    Ta        = T0+0*ΔT
    # numerics
    ny        = 127
    nx        = ceil(Int,lx_ly*ny)
    nx, ny    = nx-1, ny-1
    maxiter   = 50ny         # maximum number of pseudo-transient iterations
    nchk      = 2*ny         # error checking frequency
    nviz      = 2            # visualisation frequency
    ε_V       = 1e-8         # nonlinear absolute tolerance for momentum
    ε_∇V      = 1e-8         # nonlinear absolute tolerance for divergence
    ε_T       = 1e-10        # nonlinear absolute tolerance for divergence
    CFL       = 0.9/sqrt(2) # stability condition
    Re_mech   = 2π           # Reynolds number for Stokes problem  (numerical parameter #1)
    Re_heat   = π + sqrt(π^2 + lx^2/χ/dt)  # Reynolds number for heat conduction (numerical parameter #1)
    r         = 1.0          # Bulk to shear elastic modulus ratio (numerical parameter #2)
    nt        = 200
    # preprocessing
    dx, dy    = lx/nx, ly/ny # cell sizes
    max_lxy   = 0.5gl
    Vpdτ      = min(dx,dy)*CFL
    dτ_ρ_heat = Vpdτ*max_lxy/Re_heat/χ
    θr_dτ     = max_lxy/Vpdτ/Re_heat
    Xc, Yc    = LinRange(-(lx-dx)/2,(lx-dx)/2,nx  ), LinRange(dy/2,ly-dy/2,ny  )
    Xv, Yv    = LinRange(- lx/2    , lx/2    ,nx+1), LinRange(0   ,ly     ,ny+1)
    # allocation
    Pt        = @zeros(nx  ,ny  )
    ∇V        = @zeros(nx  ,ny  )
    τxx       = @zeros(nx  ,ny  )
    τyy       = @zeros(nx  ,ny  )
    τxy       = @zeros(nx-1,ny-1)
    Rx        = @zeros(nx-1,ny-2)
    Ry        = @zeros(nx-2,ny-1)
    RT        = @zeros(nx  ,ny  )
    Vx        = @zeros(nx+1,ny  )
    Vy        = @zeros(nx  ,ny+1)
    T         = @zeros(nx  ,ny  )
    T_o       = @zeros(nx  ,ny  )
    qTx       = @zeros(nx+1,ny  )
    qTy       = @zeros(nx  ,ny+1)
    ϕ         = @zeros(nx  ,ny  )
    μs        = μs0 .* @ones(nx  ,ny  )
    EII       = @zeros(nx  ,ny  )
    Vx_v      = copy(Vx) # visu
    Vy_v      = copy(Vy) # visu
    Pt_v      = copy(Pt) # visu
    EII_v     = copy(EII) # visu
    T_v       = copy(T)  # visu
    μs_v      = copy(μs)  # visu
    @parallel init_ϕ!(ϕ,gl,el,tanβ,ω,amp,dx,dy,lx,ly)
    @parallel init_T!(T,ϕ,T0,Ta)
    # time loop
    for it = 1:nt
        @printf("# it = %d\n", it)
        T_o .= T
        # iteration loop
        err_V=2*ε_V; err_∇V=2*ε_∇V; err_T=2*ε_T; iter=0; err_evo1=[]; err_evo2=[]
        while !((err_V <= ε_V) && (err_∇V <= ε_∇V) && (err_T <= ε_T)) && (iter <= maxiter)
            @parallel compute_P_τ_qT!(∇V, Pt, τxx, τyy, τxy, Vx, Vy, μs, qTx, qTy, T, ϕ, Vpdτ, Re_mech, r, max_lxy, χ, θr_dτ, dx, dy)
            @parallel compute_EII!(EII,Vx,Vy,ϕ,dx,dy)
            @parallel compute_V_T!(Vx, Vy, Pt, τxx, τyy, τxy, EII, μs, T, T_o, qTx, qTy, ϕ, ρgx, ρgy, μs0, T0, Ta, Q_R, dt, Vpdτ, max_lxy, Re_mech, dτ_ρ_heat, dx, dy)
            iter += 1
            if iter % nchk == 0
                @parallel compute_Res!(Rx, Ry, RT, Pt, τxx, τyy, τxy, T, T_o, qTx, qTy, EII, μs, ϕ, ρgx, ρgy, dt, dx, dy)
                norm_Rx = norm(Rx)/psc*lx/sqrt(length(Rx))
                norm_Ry = norm(Ry)/psc*lx/sqrt(length(Ry))
                norm_∇V = norm((ϕ.==fluid).*∇V)/vsc*lx/sqrt(length(∇V))
                norm_RT = norm((ϕ.==fluid).*RT)*tsc/ΔT/sqrt(length(RT))
                err_V   = maximum([norm_Rx, norm_Ry])
                err_∇V  = norm_∇V
                err_T   = norm_RT
                push!(err_evo1, maximum([norm_Rx, norm_Ry, norm_∇V, norm_RT])); push!(err_evo2,iter/ny)
                @printf("   # iters = %d, err_V = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e], err_∇V = %1.3e, err_T = %1.3e \n", iter, err_V, norm_Rx, norm_Ry, err_∇V, err_T)
            end
        end
        if it % nviz == 0
            Vx_v .= Vx; Vx_v[Vx.==0] .= NaN
            Vy_v .= Vy; Vy_v[Vy.==0] .= NaN
            Pt_v .= Pt; Pt_v[ϕ .!= fluid] .= NaN
            T_v  .= T; T_v[ϕ .!= fluid] .= NaN
            μs_v .= μs; μs_v[ϕ .!= fluid] .= NaN
            EII_v .= EII; EII_v[ϕ .!= fluid] .= NaN
            fntsz = 7
            opts  = (aspect_ratio=1, xlims=(Xv[1],Xv[end]), ylims=(Yc[1],Yc[end]), yaxis=font(fntsz,"Courier"), xaxis=font(fntsz,"Courier"), framestyle=:box, titlefontsize=fntsz, titlefont="Courier")
            opts2 = (linewidth=2, markershape=:circle, markersize=3,yaxis = (:log10, font(fntsz,"Courier")), xaxis=font(fntsz,"Courier"), framestyle=:box, titlefontsize=fntsz, titlefont="Courier")
            p1 = heatmap(Xc,Yc,Array(T_v)'; c=:batlow, title="T", opts...)
            p2 = heatmap(Xc,Yc,Array(μs_v)'; c=:batlow, title="μs", opts...)
            p3 = heatmap(Xc,Yc,Array(Pt_v)'; c=:viridis, title="Pressure", opts...)
            p4 = heatmap(Xc,Yc,Array(EII_v)'; c=:viridis, title="EII", opts...)
            p5 = heatmap(Xc,Yv,Array(Vy_v)'; c=:viridis, title="Vy", opts...)
            p6 = plot(err_evo2,err_evo1; legend=false, xlabel="# iterations/nx", ylabel="log10(error)", labels="max(error)", opts2...)
            display(plot(p1, p2, p3, p4, p5, p6, size=(1e3,900), dpi=200, layout=(3,2)))
        end
    end
    return
end

Stokes2D()
