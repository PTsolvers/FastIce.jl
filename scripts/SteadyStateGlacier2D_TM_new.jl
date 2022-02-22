const USE_GPU = haskey(ENV, "USE_GPU") ? parse(Bool, ENV["USE_GPU"]) : false
const do_save = haskey(ENV, "DO_SAVE") ? parse(Bool, ENV["DO_SAVE"]) : true
###
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Printf,Statistics,LinearAlgebra,Random
using HDF5,LightXML

@views inn(A) = A[2:end-1,2:end-1,2:end-1]
@views av(A)  = convert(eltype(A),0.5)*(A[1:end-1]+A[2:end])

include(joinpath(@__DIR__, "helpers2D.jl"))
include(joinpath(@__DIR__, "data_io2D.jl"))

import ParallelStencil: INDICES
ix,iy   = INDICES[1], INDICES[2]
ixi,iyi = :($ix+1), :($iy+1)

const air   = 0.0
const fluid = 1.0
const solid = 2.0

macro av_xii(A) esc(:( 0.5*($A[$ixi,$iyi] + $A[$ixi+1,$iyi    ]) )) end
macro av_yii(A) esc(:( 0.5*($A[$ixi,$iyi] + $A[$ixi  ,$iyi+1  ]) )) end

macro fm(A)   esc(:( $A[$ix,$iy] == fluid )) end
macro fmxy(A) esc(:( !($A[$ix,$iy] == air || $A[$ix+1,$iy] == air || $A[$ix,$iy+1] == air || $A[$ix+1,$iy+1] == air) )) end


@parallel_indices (ix,iy) function compute_EII!(EII, Vx, Vy, ϕ, dx, dy)
    nfluid = 0
    exy    = 0.0; exx = 0.0; eyy = 0.0
    if ix <= size(EII,1)-2 && iy <= size(EII,2)-2
        if ϕ[ix,iy] == fluid && ϕ[ix+1,iy] == fluid && ϕ[ix,iy+1] == fluid && ϕ[ix+1,iy+1] == fluid
            nfluid += 1
            exy    += (Vx[ix+1,iy+1] - Vx[ix+1,iy])/dy + (Vy[ix+1,iy+1] - Vy[ix,iy+1])/dx
        end
        if ϕ[ix+1,iy] == fluid && ϕ[ix+2,iy] == fluid && ϕ[ix+1,iy+1] == fluid && ϕ[ix+2,iy+1] == fluid
            nfluid += 1
            exy    += (Vx[ix+2,iy+1] - Vx[ix+2,iy])/dy + (Vy[ix+2,iy+1] - Vy[ix+1,iy+1])/dx
        end
        if ϕ[ix,iy+1] == fluid && ϕ[ix+1,iy+1] == fluid && ϕ[ix,iy+2] == fluid && ϕ[ix+1,iy+2] == fluid
            nfluid += 1
            exy    += (Vx[ix+1,iy+2] - Vx[ix+1,iy+1])/dy + (Vy[ix+1,iy+2] - Vy[ix,iy+2])/dx
        end
        if ϕ[ix+1,iy+1] == fluid && ϕ[ix+2,iy+1] == fluid && ϕ[ix+1,iy+2] == fluid && ϕ[ix+2,iy+2] == fluid
            nfluid += 1
            exy    += (Vx[ix+2,iy+2] - Vx[ix+2,iy+1])/dy + (Vy[ix+2,iy+2] - Vy[ix+1,iy+2])/dx
        end
        if nfluid > 0.0
            exy /= 2.0*nfluid
        end
        exx = (Vx[ix+2,iy+1] - Vx[ix+1,iy+1])/dx
        eyy = (Vy[ix+1,iy+2] - Vy[ix+1,iy+1])/dy
        EII[ix+1,iy+1] = (ϕ[ix+1,iy+1] == fluid)*sqrt(0.5*(exx*exx + eyy*eyy) + exy*exy)
    end
    return
end

macro Gdτ()        esc(:(vpdτ_mech*Re_mech*@all(μs)/max_lxy/(r+2.0) )) end
macro Gdτ_av()     esc(:(vpdτ_mech*Re_mech*@av(μs)/max_lxy/(r+2.0)  )) end
macro μ_veτ()      esc(:(1.0/(1.0/@Gdτ()    + 1.0/@all(μs))    )) end
macro μ_veτ_av()   esc(:(1.0/(1.0/@Gdτ_av() + 1.0/@av(μs))     )) end

@parallel function compute_P_τ_qT!(∇V, Pt, τxx, τyy, τxy, qTx, qTy, Vx, Vy, μs, ϕ, T, vpdτ_mech, Re_mech, r, max_lxy, χ, θr_dτ, dx, dy)
    @all(∇V)  = @fm(ϕ)*(@d_xa(Vx)/dx + @d_ya(Vy)/dy)
    @all(Pt)  = @fm(ϕ)*(@all(Pt) - r*@Gdτ()*@all(∇V))    
    @all(τxx) = @fm(ϕ)*2.0*@μ_veτ()*(@d_xa(Vx)/dx + @all(τxx)/@Gdτ()/2.0)
    @all(τyy) = @fm(ϕ)*2.0*@μ_veτ()*(@d_ya(Vy)/dy + @all(τyy)/@Gdτ()/2.0)
    @all(τxy) = @fmxy(ϕ)*2.0*@μ_veτ_av()*(0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx) + @all(τxy)/@Gdτ_av()/2.0)
    # thermo
    @inn_x(qTx) = (@inn_x(qTx) * θr_dτ - χ*@d_xa(T)/dx) / (θr_dτ + 1.0)
    @inn_y(qTy) = (@inn_y(qTy) * θr_dτ - χ*@d_ya(T)/dy) / (θr_dτ + 1.0)
    return
end

macro sm_xi(A) esc(:( !(($A[$ix,$iyi] == solid) || ($A[$ix+1,$iyi] == solid)) )) end
macro sm_yi(A) esc(:( !(($A[$ixi,$iy] == solid) || ($A[$ixi,$iy+1] == solid)) )) end

macro fm_xi(A) esc(:( ($A[$ix,$iyi] == fluid) && ($A[$ix+1,$iyi] == fluid) )) end
macro fm_yi(A) esc(:( ($A[$ixi,$iy] == fluid) && ($A[$ixi,$iy+1] == fluid) )) end

macro dτ_ρ_mech_ax() esc(:( vpdτ_mech*max_lxy/Re_mech/@av_xi(μs) )) end
macro dτ_ρ_mech_ay() esc(:( vpdτ_mech*max_lxy/Re_mech/@av_yi(μs) )) end

macro fa(A)   esc(:( $A[$ix,$iy] == air )) end

@parallel function compute_V_T!(Vx, Vy, T, μs, Pt, τxx, τyy, τxy, EII, T_o, qTx, qTy, ϕ, μs0, ρgx, ρgy, Ta, Q_R, T0, dt, vpdτ_mech, max_lxy, Re_mech, dτ_ρ_heat, dx, dy)
    @inn(Vx) = @sm_xi(ϕ)*( @inn(Vx) + @dτ_ρ_mech_ax()*(@d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(Pt)/dx - @fm_xi(ϕ)*ρgx) )
    @inn(Vy) = @sm_yi(ϕ)*( @inn(Vy) + @dτ_ρ_mech_ay()*(@d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(Pt)/dy - @fm_yi(ϕ)*ρgy) )
    # thermo
    @all(T)  = (@all(T) + dτ_ρ_heat*(@all(T_o)/dt - @d_xa(qTx)/dx - @d_ya(qTy)/dy + 2.0*@all(μs)*@all(EII)))/(1.0 + dτ_ρ_heat/dt)
    @all(μs) = μs0*exp(-Q_R*(1.0 - T0/@all(T)))
    return
end

@parallel function compute_Res!(Rx, Ry, RT, Pt, τxx, τyy, τxy, T, T_o, qTx, qTy, EII, μs, ϕ, ρgx, ρgy, dt, dx, dy)
    @all(Rx) = @sm_xi(ϕ)*(@d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(Pt)/dx - @fm_xi(ϕ)*ρgx)
    @all(Ry) = @sm_yi(ϕ)*(@d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(Pt)/dy - @fm_yi(ϕ)*ρgy)
    # thermo
    @all(RT) = -(@all(T) - @all(T_o))/dt - (@d_xa(qTx)/dx + @d_ya(qTy)/dy) + 2.0*@all(μs)*@all(EII)
    return
end

@parallel function preprocess_visu!(Vn, τII, Ptv, EIIv, Tv, μsv, Vx, Vy, τxx, τyy, τxy, Pt, EII, T, μs)
    # all arrays of size (nx-2,ny-2)
    @all(Vn)   = sqrt(@av_xii(Vx)*@av_xii(Vx) + @av_yii(Vy)*@av_yii(Vy))
    @all(τII)  = sqrt(0.5*(@inn(τxx)*@inn(τxx) + @inn(τyy)*@inn(τyy)) + @av(τxy)*@av(τxy))
    @all(Ptv)  = @inn(Pt)
    @all(EIIv) = @inn(EII)
    @all(Tv)   = @inn(T)
    @all(μsv)  = @inn(μs)
    return
end

@parallel_indices (ix,iy) function apply_mask!(Vn, τII, Ptv, EIIv, Tv, μsv, ϕ)
    if checkbounds(Bool,Vn,ix,iy)
        if ϕ[ix+1,iy+1] != fluid
            Vn[ix,iy]   = NaN
            Ptv[ix,iy]  = NaN
            τII[ix,iy]  = NaN
            EIIv[ix,iy] = NaN
            # Tv[ix,iy]   = NaN
            μsv[ix,iy]  = NaN
        end
    end
    return
end

"Check if index is inside phase."
function is_inside_phase(y3rot,ytopo)
    return y3rot < ytopo
end

@parallel_indices (ix,iy) function set_phases!(ϕ,ysurf,ybed,R,ox,oy,osx,dx,dy,dsx)
    if checkbounds(Bool,ϕ,ix,iy)
        xc,yc   = ox + (ix-1)*dx, oy + (iy-1)*dy
        xrot    = R[1,1]*xc + R[1,2]*yc
        yrot    = R[2,1]*xc + R[2,2]*yc
        ixr     = clamp(floor(Int, (xrot-osx)/dsx) + 1, 1, size(ysurf,1))
        if is_inside_phase(yrot,ysurf[ixr])
            ϕ[ix,iy] = fluid
        end
        if is_inside_phase(yrot,ybed[ixr])
            ϕ[ix,iy] = solid
        end
    end
    return
end

@parallel_indices (ix,iy) function init_ϕi!(ϕ,ϕx,ϕy)
    if ix <= size(ϕx,1) && iy <= size(ϕx,2)
        ϕx[ix,iy] = air
        if ϕ[ix,iy] == fluid && ϕ[ix+1,iy] == fluid
            ϕx[ix,iy] = fluid
        end
    end
    if ix <= size(ϕy,1) && iy <= size(ϕy,2)
        ϕy[ix,iy] = air
        if ϕ[ix,iy] == fluid && ϕ[ix,iy+1] == fluid
            ϕy[ix,iy] = fluid
        end
    end
    return
end

@views function Stokes2D(dem)
    # inputs
    nx,ny    = 63,63         # local resolution
    nt       = 100           # number of time steps
    ns       = 2             # number of oversampling per cell
    out_path = "../out_visu"
    out_name = "results2D"
    nsave    = 10
    # define domain
    domain   = dilate(rotated_domain(dem), (0.05, 0.05))
    lx,ly    = extents(domain)
    xv,yv    = create_grid(domain,(nx+1,ny+1))
    xc,yc    = av.((xv,yv))
    dx,dy    = lx/nx,ly/ny
    R        = rotation(dem)
    # physics
    ## dimensionally independent
    μs0      = 1.0  # matrix viscosity [Pa*s]
    ρg0      = 1.0  # gravity          [Pa/m]
    ΔT       = 1.0  # temperature difference between ice and atmosphere [K]
    ## scales
    psc      = ρg0*ly
    tsc      = μs0/psc
    vsc      = ly/tsc
    # nondimensional
    T0_δT    = 1.0
    Q_R      = 3.0e1
    ## dimensionally dependent
    ρgv      = ρg0*R'*[0,1]
    ρgx,ρgy  = ρgv
    χ        = 1e-3*ly^2/tsc # m^2/s = ly^3 * ρg0 / μs0
    T0       = T0_δT*ΔT
    dt       = 1e-2*tsc
    Ta       = T0+0*ΔT
    # numerics
    maxiter  = 50ny         # maximum number of pseudo-transient iterations
    nchk     = 2*ny         # error checking frequency
    ε_V      = 1e-6         # nonlinear absolute tolerance for momentum
    ε_∇V     = 1e-6        # nonlinear absolute tolerance for divergence
    ε_T      = 1e-8         # nonlinear absolute tolerance for temperature
    CFL_mech = 0.8/sqrt(2)  # stability condition
    CFL_heat = 0.95/sqrt(2) # stability condition
    Re_mech  = 2π           # Reynolds number                     (numerical parameter #1)
    r_mech   = 1.0          # Bulk to shear elastic modulus ratio (numerical parameter #2)
    Re_heat  = π + sqrt(π^2 + lx^2/χ/dt)  # Reynolds number for heat conduction (numerical parameter #1)
    # preprocessing
    max_lxy = 0.25ly
    vpdτ_mech = min(dx,dy)*CFL_mech
    vpdτ_heat = min(dx,dy)*CFL_heat
    dτ_ρ_heat = vpdτ_heat*max_lxy/Re_heat/χ
    θr_dτ     = max_lxy/vpdτ_heat/Re_heat
    # allocation
    Pt       = @zeros(nx  ,ny  )
    ∇V       = @zeros(nx  ,ny  )
    τxx      = @zeros(nx  ,ny  )
    τyy      = @zeros(nx  ,ny  )
    τxy      = @zeros(nx-1,ny-1)
    Rx       = @zeros(nx-1,ny-2)
    Ry       = @zeros(nx-2,ny-1)
    RT       = @zeros(nx  ,ny  )
    ϕx       = @zeros(nx-1,ny-2)
    ϕy       = @zeros(nx-2,ny-1)
    Vx       = @zeros(nx+1,ny  )
    Vy       = @zeros(nx  ,ny+1)
    EII      = @zeros(nx  ,ny  )
    T_o      = @zeros(nx  ,ny  )
    qTx      = @zeros(nx+1,ny  )
    qTy      = @zeros(nx  ,ny+1)
    μs       = μs0 .* @ones(nx  ,ny  )
    T        = T0  .* @ones(nx  ,ny  )
    # set phases
    print("Set phases (0-air, 1-ice, 2-bedrock)...")
    Rinv     = Data.Array(R')
    # supersampled grid
    nr_box       = dem.domain
    xc_ss        = LinRange(nr_box.xmin,nr_box.xmax,ns*length(xc))
    dsx          = xc_ss[2] - xc_ss[1]
    y_bed,y_surf = Data.Array.(evaluate(dem, xc_ss))
    ϕ            = air.*@ones(nx,ny)
    @parallel set_phases!(ϕ,y_surf,y_bed,Rinv,xc[1],yc[1],xc_ss[1],dx,dy,dsx)
    @parallel init_ϕi!(ϕ, ϕx, ϕy)
    len_g = sum(ϕ.==fluid)
    # visu
    if do_save
        !ispath(out_path) && mkdir(out_path)
        Vn   = @zeros(nx-2,ny-2)
        τII  = @zeros(nx-2,ny-2)
        Ptv  = @zeros(nx-2,ny-2)
        EIIv = @zeros(nx-2,ny-2)
        Tv   = @zeros(nx-2,ny-2)
        μsv  = @zeros(nx-2,ny-2)
    end
    println(" done. Starting the real stuff 😎")
    # time loop
    for it = 1:nt
        @printf("# it = %d\n", it)
        T_o .= T
        # iteration loop
        err_V=2*ε_V; err_∇V=2*ε_∇V; err_T=2*ε_T; iter=0; err_evo1=[]; err_evo2=[]
        while !((err_V <= ε_V) && (err_∇V <= ε_∇V) && (err_T <= ε_T)) && (iter <= maxiter)
            @parallel compute_EII!(EII, Vx, Vy, ϕ, dx, dy)
            @parallel compute_P_τ_qT!(∇V, Pt, τxx, τyy, τxy, qTx, qTy, Vx, Vy, μs, ϕ, T, vpdτ_mech, Re_mech, r_mech, max_lxy, χ, θr_dτ, dx, dy)
            @parallel compute_V_T!(Vx, Vy, T, μs, Pt, τxx, τyy, τxy, EII, T_o, qTx, qTy, ϕ, μs0, ρgx, ρgy, Ta, Q_R, T0, dt, vpdτ_mech, max_lxy, Re_mech, dτ_ρ_heat, dx, dy)
            iter += 1
            if iter % nchk == 0
                @parallel compute_Res!(Rx, Ry, RT, Pt, τxx, τyy, τxy, T, T_o, qTx, qTy, EII, μs, ϕ, ρgx, ρgy, dt, dx, dy)
                norm_Rx = norm(Rx)/psc*ly/sqrt(len_g)
                norm_Ry = norm(Ry)/psc*ly/sqrt(len_g)
                norm_∇V = norm(∇V)/vsc*ly/sqrt(len_g)
                norm_T  = norm(RT)*tsc/ΔT/sqrt(len_g)
                err_V   = maximum([norm_Rx, norm_Ry])
                err_∇V  = norm_∇V
                err_T   = norm_T
                # push!(err_evo1, maximum([norm_Rx, norm_Ry, norm_∇V])); push!(err_evo2,iter/nx)
                @printf("# iters = %d, err_V = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e], err_∇V = %1.3e, err_T = %1.3e \n", iter, err_V, norm_Rx, norm_Ry, err_∇V, err_T)
            end
        end
        if do_save && (it % nsave == 0)
            @parallel preprocess_visu!(Vn, τII, Ptv, EIIv, Tv, μsv, Vx, Vy, τxx, τyy, τxy, Pt, EII, T, μs)
            @parallel apply_mask!(Vn, τII, Ptv, EIIv, Tv, μsv, ϕ)
            out_h5 = joinpath(out_path,out_name)*".h5"
            I = CartesianIndices(( 1:nx-2, 1:ny-2 ))
            fields = Dict("Phi"=>ϕ[2:end-1,2:end-1],"Vn"=>Vn,"τII"=>τII,"Pr"=>Ptv,"EII"=>EIIv,"T"=>Tv,"μ"=>μsv)
            print("Saving HDF5 file...")
            write_h5(out_h5,fields,(nx-2,ny-2),I) # comm_cart,MPI.Info() are varargs
            println(" done")
            # write XDMF
            print("Saving XDMF file...")
            write_xdmf(joinpath(out_path,out_name)*".xdmf3",out_h5,fields,(xc[1],yc[1]),(dx,dy),(nx-2,ny-2))
            println(" done")
        end
    end
    return
end

# Stokes3D(load_elevation("../data/alps/data_Rhone.h5"))
Stokes2D(generate_elevation(2.0,(-0.25,0.82),1/25,10π,tan(-π/12),0.1,0.9))
