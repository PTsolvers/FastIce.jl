const USE_GPU = haskey(ENV, "USE_GPU") ? parse(Bool, ENV["USE_GPU"]) : true
const do_save = haskey(ENV, "DO_SAVE") ? parse(Bool, ENV["DO_SAVE"]) : true
###
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end
using ImplicitGlobalGrid,Printf,Statistics,LinearAlgebra,Random
import MPI
using HDF5,LightXML

norm_g(A) = (sum2_l = sum(A.^2); sqrt(MPI.Allreduce(sum2_l, MPI.SUM, MPI.COMM_WORLD)))
sum_g(A)  = (sum_l  = sum(A); MPI.Allreduce(sum_l, MPI.SUM, MPI.COMM_WORLD))
max_g(A)  = (max_l  = maximum(A); MPI.Allreduce(max_l, MPI.MAX, MPI.COMM_WORLD))

@views inn(A) = A[2:end-1,2:end-1,2:end-1]
@views av(A)  = convert(eltype(A),0.5)*(A[1:end-1]+A[2:end])
@views function smooth2D!(A2, A, fact)
    A2[2:end-1,2:end-1] .= A[2:end-1,2:end-1] .+ 1.0./4.1./fact.*( (A[3:end,2:end-1].-2.0.*A[2:end-1,2:end-1].+A[1:end-2,2:end-1]).+(A[2:end-1,3:end].-2.0.*A[2:end-1,2:end-1].+A[2:end-1,1:end-2]) )
    return
end

include(joinpath(@__DIR__,"helpers3D_v5.jl"))
include(joinpath(@__DIR__,"data_io.jl"))
include(joinpath(@__DIR__,"flagging_macros3D.jl"))

const air   = 0.0
const fluid = 1.0
const solid = 2.0

macro av_xii(A) esc(:( 0.5*($A[$ixi,$iyi,$izi] + $A[$ixi+1,$iyi  ,$izi  ]) )) end
macro av_yii(A) esc(:( 0.5*($A[$ixi,$iyi,$izi] + $A[$ixi  ,$iyi+1,$izi  ]) )) end
macro av_zii(A) esc(:( 0.5*($A[$ixi,$iyi,$izi] + $A[$ixi  ,$iyi  ,$izi+1]) )) end

@parallel_indices (ix,iy,iz) function compute_EII!(EII, Vx, Vy, Vz, ϕ, dx, dy, dz)
    nfluid_xy = 0; nfluid_xz = 0; nfluid_yz = 0; 
    exy = 0.0; exz = 0.0; eyz = 0.0; exx = 0.0; eyy = 0.0; ezz = 0.0
    if ix <= size(EII,1)-2 && iy <= size(EII,2)-2 && iz <= size(EII,3)-2
        if ϕ[ix,iy,iz+1] == fluid && ϕ[ix+1,iy,iz+1] == fluid && ϕ[ix,iy+1,iz+1] == fluid && ϕ[ix+1,iy+1,iz+1] == fluid
            nfluid_xy += 1
            exy += (Vx[ix+1,iy+1,iz+1] - Vx[ix+1,iy,iz+1])/dy + (Vy[ix+1,iy+1,iz+1] - Vy[ix,iy+1,iz+1])/dx
        end
        if ϕ[ix+1,iy,iz+1] == fluid && ϕ[ix+2,iy,iz+1] == fluid && ϕ[ix+1,iy+1,iz+1] == fluid && ϕ[ix+2,iy+1,iz+1] == fluid
            nfluid_xy += 1
            exy += (Vx[ix+2,iy+1,iz+1] - Vx[ix+2,iy,iz+1])/dy + (Vy[ix+2,iy+1,iz+1] - Vy[ix+1,iy+1,iz+1])/dx
        end
        if ϕ[ix,iy+1,iz+1] == fluid && ϕ[ix+1,iy+1,iz+1] == fluid && ϕ[ix,iy+2,iz+1] == fluid && ϕ[ix+1,iy+2,iz+1] == fluid
            nfluid_xy += 1
            exy += (Vx[ix+1,iy+2,iz+1] - Vx[ix+1,iy+1,iz+1])/dy + (Vy[ix+1,iy+2,iz+1] - Vy[ix,iy+2,iz+1])/dx
        end
        if ϕ[ix+1,iy+1,iz+1] == fluid && ϕ[ix+2,iy+1,iz+1] == fluid && ϕ[ix+1,iy+2,iz+1] == fluid && ϕ[ix+2,iy+2,iz+1] == fluid
            nfluid_xy += 1
            exy += (Vx[ix+2,iy+2,iz+1] - Vx[ix+2,iy+1,iz+1])/dy + (Vy[ix+2,iy+2,iz+1] - Vy[ix+1,iy+2,iz+1])/dx
        end
        # ----------------------------------------------------------------------------------------------------
        if ϕ[ix,iy+1,iz] == fluid && ϕ[ix+1,iy+1,iz] == fluid && ϕ[ix,iy+1,iz+1] == fluid && ϕ[ix+1,iy+1,iz+1] == fluid
            nfluid_xz += 1
            exz += (Vx[ix+1,iy+1,iz+1] - Vx[ix+1,iy+1,iz])/dz + (Vz[ix+1,iy+1,iz+1] - Vz[ix,iy+1,iz+1])/dx
        end
        if ϕ[ix+1,iy+1,iz] == fluid && ϕ[ix+2,iy+1,iz] == fluid && ϕ[ix+1,iy+1,iz+1] == fluid && ϕ[ix+2,iy+1,iz+1] == fluid
            nfluid_xz += 1
            exz += (Vx[ix+2,iy+1,iz+1] - Vx[ix+2,iy+1,iz])/dz + (Vz[ix+2,iy+1,iz+1] - Vz[ix+1,iy+1,iz+1])/dx
        end
        if ϕ[ix,iy+1,iz+1] == fluid && ϕ[ix+1,iy+1,iz+1] == fluid && ϕ[ix,iy+1,iz+2] == fluid && ϕ[ix+1,iy+1,iz+2] == fluid
            nfluid_xz += 1
            exz += (Vx[ix+1,iy+1,iz+2] - Vx[ix+1,iy+1,iz+1])/dz + (Vz[ix+1,iy+1,iz+2] - Vz[ix,iy+1,iz+2])/dx
        end
        if ϕ[ix+1,iy+1,iz+1] == fluid && ϕ[ix+2,iy+1,iz+1] == fluid && ϕ[ix+1,iy+1,iz+2] == fluid && ϕ[ix+2,iy+1,iz+2] == fluid
            nfluid_xz += 1
            exz += (Vx[ix+2,iy+1,iz+2] - Vx[ix+2,iy+1,iz+1])/dz + (Vz[ix+2,iy+1,iz+2] - Vz[ix+1,iy+1,iz+2])/dx
        end
        # ----------------------------------------------------------------------------------------------------
        if ϕ[ix+1,iy,iz] == fluid && ϕ[ix+1,iy+1,iz] == fluid && ϕ[ix+1,iy,iz+1] == fluid && ϕ[ix+1,iy+1,iz+1] == fluid
            nfluid_yz += 1
            eyz += (Vy[ix+1,iy+1,iz+1] - Vy[ix+1,iy+1,iz])/dz + (Vz[ix+1,iy+1,iz+1] - Vz[ix+1,iy,iz+1])/dy
        end
        if ϕ[ix+1,iy+1,iz] == fluid && ϕ[ix+1,iy+2,iz] == fluid && ϕ[ix+1,iy+1,iz+1] == fluid && ϕ[ix+1,iy+2,iz+1] == fluid
            nfluid_yz += 1
            eyz += (Vy[ix+1,iy+2,iz+1] - Vy[ix+1,iy+2,iz])/dz + (Vz[ix+1,iy+2,iz+1] - Vz[ix+1,iy+1,iz+1])/dy
        end
        if ϕ[ix+1,iy,iz+1] == fluid && ϕ[ix+1,iy+1,iz+1] == fluid && ϕ[ix+1,iy,iz+2] == fluid && ϕ[ix+1,iy+1,iz+2] == fluid
            nfluid_yz += 1
            eyz += (Vy[ix+1,iy+1,iz+2] - Vy[ix+1,iy+1,iz+1])/dz + (Vz[ix+1,iy+1,iz+2] - Vz[ix+1,iy,iz+2])/dy
        end
        if ϕ[ix+1,iy+1,iz+1] == fluid && ϕ[ix+1,iy+2,iz+1] == fluid && ϕ[ix+1,iy+1,iz+2] == fluid && ϕ[ix+1,iy+2,iz+2] == fluid
            nfluid_yz += 1
            eyz += (Vy[ix+1,iy+2,iz+2] - Vy[ix+1,iy+2,iz+1])/dz + (Vz[ix+1,iy+2,iz+2] - Vz[ix+1,iy+1,iz+2])/dy
        end
        if (nfluid_xy > 0.0)  exy /= 2.0*nfluid_xy  end
        if (nfluid_xz > 0.0)  exz /= 2.0*nfluid_xz  end
        if (nfluid_yz > 0.0)  eyz /= 2.0*nfluid_yz  end
        exx = (Vx[ix+2,iy+1,iz+1] - Vx[ix+1,iy+1,iz+1])/dx
        eyy = (Vy[ix+1,iy+2,iz+1] - Vy[ix+1,iy+1,iz+1])/dy
        ezz = (Vz[ix+1,iy+1,iz+2] - Vz[ix+1,iy+1,iz+1])/dz
        EII[ix+1,iy+1,iz+1] = (ϕ[ix+1,iy+1,iz+1] == fluid)*sqrt(0.5*(exx*exx + eyy*eyy + ezz*ezz) + exy*exy + exz*exz + eyz*eyz)
    end
    return
end

macro Gdτ()          esc(:( vpdτ_mech*Re_mech*@all(μs)/max_lxyz/(r+2.0)    )) end
macro Gdτ_av_xyi()   esc(:( vpdτ_mech*Re_mech*@av_xyi(μs)/max_lxyz/(r+2.0) )) end
macro Gdτ_av_xzi()   esc(:( vpdτ_mech*Re_mech*@av_xzi(μs)/max_lxyz/(r+2.0) )) end
macro Gdτ_av_yzi()   esc(:( vpdτ_mech*Re_mech*@av_yzi(μs)/max_lxyz/(r+2.0) )) end
macro μ_veτ()        esc(:( 1.0/(1.0/@Gdτ()        + 1.0/@all(μs))         )) end
macro μ_veτ_av_xyi() esc(:( 1.0/(1.0/@Gdτ_av_xyi() + 1.0/@av_xyi(μs))      )) end
macro μ_veτ_av_xzi() esc(:( 1.0/(1.0/@Gdτ_av_xzi() + 1.0/@av_xzi(μs))      )) end
macro μ_veτ_av_yzi() esc(:( 1.0/(1.0/@Gdτ_av_yzi() + 1.0/@av_yzi(μs))      )) end

@parallel_indices (ix,iy,iz) function compute_P_τ_qT!(∇V, Pt, τxx, τyy, τzz, τxy, τxz, τyz, qTx, qTy, qTz, Vx, Vy, Vz, μs, ϕ, T, vpdτ_mech, Re_mech, r, max_lxyz, χ, θr_dτ, dx, dy, dz)
    @define_indices ix iy iz
    @in_phase ϕ fluid begin
        @all(∇V)  = @d_xa(Vx)/dx + @d_ya(Vy)/dy + @d_za(Vz)/dz
        @all(Pt)  = @all(Pt) - r*@Gdτ()*@all(∇V)
        @all(τxx) = 2.0*@μ_veτ()*(@d_xa(Vx)/dx + @all(τxx)/@Gdτ()/2.0)
        @all(τyy) = 2.0*@μ_veτ()*(@d_ya(Vy)/dy + @all(τyy)/@Gdτ()/2.0)
        @all(τzz) = 2.0*@μ_veτ()*(@d_za(Vz)/dz + @all(τzz)/@Gdτ()/2.0)
    end
    @corner_xy ϕ air fluid begin @all(τxy) = 2.0*@μ_veτ_av_xyi()*(0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx) + @all(τxy)/@Gdτ_av_xyi()/2.0) end
    @corner_xz ϕ air fluid begin @all(τxz) = 2.0*@μ_veτ_av_xzi()*(0.5*(@d_zi(Vx)/dz + @d_xi(Vz)/dx) + @all(τxz)/@Gdτ_av_xzi()/2.0) end
    @corner_yz ϕ air fluid begin @all(τyz) = 2.0*@μ_veτ_av_yzi()*(0.5*(@d_zi(Vy)/dz + @d_yi(Vz)/dy) + @all(τyz)/@Gdτ_av_yzi()/2.0) end
    # thermo
    @within_x ϕ begin @inn_x(qTx) = (@inn_x(qTx) * θr_dτ - χ*@d_xa(T)/dx) / (θr_dτ + 1.0) end
    @within_y ϕ begin @inn_y(qTy) = (@inn_y(qTy) * θr_dτ - χ*@d_ya(T)/dy) / (θr_dτ + 1.0) end
    @within_z ϕ begin @inn_z(qTz) = (@inn_z(qTz) * θr_dτ - χ*@d_za(T)/dz) / (θr_dτ + 1.0) end
    return
end

macro fm_xi(A) esc(:( !(($A[$ix,$iyi,$izi] == air) && ($A[$ix+1,$iyi,$izi] == air)) )) end
macro fm_yi(A) esc(:( !(($A[$ixi,$iy,$izi] == air) && ($A[$ixi,$iy+1,$izi] == air)) )) end
macro fm_zi(A) esc(:( !(($A[$ixi,$iyi,$iz] == air) && ($A[$ixi,$iyi,$iz+1] == air)) )) end

macro dτ_ρ_mech_ax() esc(:( vpdτ_mech*max_lxyz/Re_mech/@av_xi(μs) )) end
macro dτ_ρ_mech_ay() esc(:( vpdτ_mech*max_lxyz/Re_mech/@av_yi(μs) )) end
macro dτ_ρ_mech_az() esc(:( vpdτ_mech*max_lxyz/Re_mech/@av_zi(μs) )) end

@parallel_indices (ix,iy,iz) function compute_V_T_μ!(Vx, Vy, Vz, T, μs, Pt, τxx, τyy, τzz, τxy, τxz, τyz, EII, T_o, qTx, qTy, qTz, ϕ, μs0, ρgx, ρgy, ρgz, Ta, Q_R, T0, dt, npow, γ, vpdτ_mech, max_lxyz, Re_mech, dτ_ρ_heat, dx, dy, dz)
    @define_indices ix iy iz
    @not_in_phases_xi ϕ solid solid begin @inn(Vx) = @inn(Vx) + @dτ_ρ_mech_ax()*(@d_xi(τxx)/dx + @d_ya(τxy)/dy + @d_za(τxz)/dz - @d_xi(Pt)/dx - @fm_xi(ϕ)*ρgx) end
    @not_in_phases_yi ϕ solid solid begin @inn(Vy) = @inn(Vy) + @dτ_ρ_mech_ay()*(@d_yi(τyy)/dy + @d_xa(τxy)/dx + @d_za(τyz)/dz - @d_yi(Pt)/dy - @fm_yi(ϕ)*ρgy) end
    @not_in_phases_zi ϕ solid solid begin @inn(Vz) = @inn(Vz) + @dτ_ρ_mech_az()*(@d_zi(τzz)/dz + @d_xa(τxz)/dx + @d_ya(τyz)/dy - @d_zi(Pt)/dz - @fm_zi(ϕ)*ρgz) end
    # thermo
    @for_all ϕ begin @all(T)  = (@all(T) + dτ_ρ_heat*(@all(T_o)/dt - @d_xa(qTx)/dx - @d_ya(qTy)/dy - @d_za(qTz)/dz + 2.0*@all(μs)*@all(EII)))/(1.0 + dτ_ρ_heat/dt) end
    @for_all ϕ begin @all(μs) = (1.0-γ)*@all(μs) + γ*(( @all(EII)^(1.0/npow-1.0) * exp(-Q_R*(1.0 - T0/@all(T))) )^(-1) + 1.0/μs0)^(-1) end
    return
end

@parallel_indices (ix,iy,iz) function compute_Res!(Rx, Ry, Rz, RT, Pt, τxx, τyy, τzz, τxy, τxz, τyz, T, T_o, qTx, qTy, qTz, EII, μs, ϕ, ρgx, ρgy, ρgz, dt, dx, dy, dz)
    @define_indices ix iy iz
    @not_in_phases_xi ϕ solid solid begin @all(Rx) = @d_xi(τxx)/dx + @d_ya(τxy)/dy + @d_za(τxz)/dz - @d_xi(Pt)/dx - @fm_xi(ϕ)*ρgx end
    @not_in_phases_yi ϕ solid solid begin @all(Ry) = @d_yi(τyy)/dy + @d_xa(τxy)/dx + @d_za(τyz)/dz - @d_yi(Pt)/dy - @fm_yi(ϕ)*ρgy end
    @not_in_phases_zi ϕ solid solid begin @all(Rz) = @d_zi(τzz)/dz + @d_xa(τxz)/dx + @d_ya(τyz)/dy - @d_zi(Pt)/dz - @fm_zi(ϕ)*ρgz end
    # thermo
    @for_all ϕ begin @all(RT) = -(@all(T) - @all(T_o))/dt - (@d_xa(qTx)/dx + @d_ya(qTy)/dy + @d_za(qTz)/dz) + 2.0*@all(μs)*@all(EII) end
    return
end

@parallel function preprocess_visu!(Vn, τII, Vx, Vy, Vz, τxx, τyy, τzz, τxy, τxz, τyz)
    # all arrays of size (nx-2,ny-2,nz-2)
    @all(Vn)  = sqrt(@av_xii(Vx)*@av_xii(Vx) + @av_yii(Vy)*@av_yii(Vy) + @av_zii(Vz)*@av_zii(Vz))
    @all(τII) = sqrt(0.5*(@inn(τxx)*@inn(τxx) + @inn(τyy)*@inn(τyy) + @inn(τzz)*@inn(τzz)) + @av_xya(τxy)*@av_xya(τxy) + @av_xza(τxz)*@av_xza(τxz) + @av_yza(τyz)*@av_yza(τyz))
    return
end

@parallel_indices (ix,iy,iz) function apply_mask!(Vn, τII, ϕ)
    if checkbounds(Bool,Vn,ix,iy,iz)
        if ϕ[ix+1,iy+1,iz+1] != fluid
            Vn[ix,iy,iz]  = NaN
            τII[ix,iy,iz] = NaN
        end
    end
    return
end

"Check if index is inside phase."
function is_inside_phase(z3rot,ztopo)
    return z3rot < ztopo
end

@parallel_indices (ix,iy,iz) function set_phases!(ϕ,zsurf,zbed,R,ox,oy,oz,osx,osy,dx,dy,dz,dsx,dsy)
    if checkbounds(Bool,ϕ,ix,iy,iz)
        xc,yc,zc    = ox + (ix-1)*dx, oy + (iy-1)*dy, oz + (iz-1)*dz
        xrot        = R[1,1]*xc + R[1,2]*yc + R[1,3]*zc
        yrot        = R[2,1]*xc + R[2,2]*yc + R[2,3]*zc
        zrot        = R[3,1]*xc + R[3,2]*yc + R[3,3]*zc
        ixr         = clamp(floor(Int, (xrot-osx)/dsx) + 1, 1, size(zsurf,1))
        iyr         = clamp(floor(Int, (yrot-osy)/dsy) + 1, 1, size(zsurf,2))
        if is_inside_phase(zrot,zsurf[ixr,iyr])
            ϕ[ix,iy,iz] = fluid
        end
        if is_inside_phase(zrot,zbed[ixr,iyr])
            ϕ[ix,iy,iz] = solid
        end
    end
    return
end

@parallel_indices (iy,iz) function bc_x!(A)
    A[  1, iy,  iz] = A[    2,   iy,   iz]
    A[end, iy,  iz] = A[end-1,   iy,   iz]
    return
end

@parallel_indices (ix,iz) function bc_y!(A)
    A[ ix,  1,  iz] = A[   ix,    2,   iz]
    A[ ix,end,  iz] = A[   ix,end-1,   iz]
    return
end

@parallel_indices (ix,iy) function bc_z!(A)
    A[ ix,  iy,  1] = A[   ix,   iy,    2]
    A[ ix,  iy,end] = A[   ix,   iy,end-1]
    return
end

@views function Stokes3D(dem)
    # inputs
    # nx,ny,nz  = 511,511,383      # local resolution
    # nx,ny,nz  = 127,127,95       # local resolution
    nx,ny,nz  = 127,255,95       # local resolution
    dim       = (2,2,1)          # MPI dims
    # nx,ny,nz  = 255,511,95       # local resolution
    # nx,ny,nz  = 95,127,47         # local resolution
    # dim       = (1,1,1)          # MPI dims
    nt        = 1                # number of timesteps
    ns        = 4                # number of oversampling per cell
    nsm       = 2                # number of surface data smoothing steps
    out_path  = "../out_visu"
    out_name  = "results3D_TM_rhone"
    # out_name  = "results3D_TM_lowres"
    nsave     = 1
    # IGG initialisation
    me,dims,nprocs,coords,comm_cart = init_global_grid(nx,ny,nz;dimx=dim[1],dimy=dim[2],dimz=dim[3])
    info      = MPI.Info()
    # define domain
    domain    = dilate(rotated_domain(dem), (0.05, 0.05, 0.05))
    lx,ly,lz  = extents(domain)
    xv,yv,zv  = create_grid(domain,(nx_g()+1,ny_g()+1,nz_g()+1))
    xc,yc,zc  = av.((xv,yv,zv))
    dx,dy,dz  = lx/nx_g(),ly/ny_g(),lz/nz_g()
    R         = rotation(dem)
    (me==0) && println("lx, ly, lz = $lx, $ly, $lz")
    (me==0) && println("dx, dy, dz = $dx, $dy, $dz")
    # physics
    ## dimensionally independent
    μs0       = 0.1#1.0               # matrix viscosity [Pa*s]
    ρg0       = 1.0               # gravity          [Pa/m]
    ΔT        = 1.0               # temperature difference between ice and atmosphere [K]
    npow      = 3.0
    ## scales
    psc       = ρg0*lz
    tsc       = μs0/psc
    vsc       = lz/tsc
    # nondimensional
    T0_δT     = 1.0
    Q_R       = 3.0e1
    ## dimensionally dependent
    ρgv       = ρg0*R'*[0,0,1]
    ρgx,ρgy,ρgz = ρgv
    χ         = 1e-2*ly^2/tsc # m^2/s = ly^3 * ρg0 / μs0
    T0        = T0_δT*ΔT
    dt        = 5e-3*tsc
    Ta        = T0+0*ΔT
    # numerics
    maxiter   = 30*nx_g()     # maximum number of pseudo-transient iterations
    nchk      = 2*nx_g()      # error checking frequency
    b_width   = (8,4,4)       # boundary width
    γ         = 1e-1
    ε_V       = 1e-6          # nonlinear absolute tolerance for momentum
    ε_∇V      = 1e-6          # nonlinear absolute tolerance for divergence
    ε_T       = 1e-8          # nonlinear absolute tolerance for temperature
    CFL_mech  = 0.5/sqrt(3)   # stability condition
    CFL_heat  = 0.95/sqrt(3)  # stability condition
    Re_mech   = 2π            # Reynolds number                     (numerical parameter #1)
    r_mech    = 1.0           # Bulk to shear elastic modulus ratio (numerical parameter #2)
    Re_heat   = π + sqrt(π^2 + lx^2/χ/dt)  # Reynolds number for heat conduction (numerical parameter #1)
    # preprocessing
    max_lxyz  = 0.35lz
    vpdτ_mech = min(dx,dy,dz)*CFL_mech
    vpdτ_heat = min(dx,dy,dz)*CFL_heat
    dτ_ρ_heat = vpdτ_heat*max_lxyz/Re_heat/χ
    θr_dτ     = max_lxyz/vpdτ_heat/Re_heat
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
    RT        = @zeros(nx  ,ny  ,nz  )
    Vx        = @zeros(nx+1,ny  ,nz  )
    Vy        = @zeros(nx  ,ny+1,nz  )
    Vz        = @zeros(nx  ,ny  ,nz+1)
    EII       = @zeros(nx  ,ny  ,nz  )
    T_o       = @zeros(nx  ,ny  ,nz  )
    qTx       = @zeros(nx+1,ny  ,nz  )
    qTy       = @zeros(nx  ,ny+1,nz  )
    qTz       = @zeros(nx  ,ny  ,nz+1)
    μs        = 1e0μs0 .* @ones(nx,ny,nz)
    T         =    T0  .* @ones(nx,ny,nz)
    # set phases
    if (me==0) print("Set phases (0-air, 1-ice, 2-bedrock)...") end
    # local dem eval
    xc_l,yc_l,zc_l = local_grid(xc,yc,zc,nx,ny,nz,coords)
    xcl_min,xcl_max,ycl_min,ycl_max = local_extend(xc,yc,zc,nx,ny,nz,dx,dy,coords,R)
    nr_box = dem.domain
    xcl_min,xcl_max,ycl_min,ycl_max = max(xcl_min,nr_box.xmin),min(xcl_max,nr_box.xmax),max(ycl_min,nr_box.ymin),min(ycl_max,nr_box.ymax)
    # supersampled grid
    xc_ss,yc_ss  = LinRange(xcl_min,xcl_max,ns*length(xc_l)),LinRange(ycl_min,ycl_max,ns*length(yc_l))
    dsx,dsy      = xc_ss[2] - xc_ss[1], yc_ss[2] - yc_ss[1]
    z_bed,z_surf = Data.Array.(evaluate(dem, xc_ss, yc_ss))
    ϕ            = air.*@ones(nx,ny,nz)
    z_bed2  = copy(z_bed)
    z_surf2 = copy(z_surf)
    for ism = 1:nsm
        smooth2D!(z_bed2, z_bed, 1.0)
        smooth2D!(z_surf2, z_surf, 1.0)
        z_bed, z_bed2 = z_bed2, z_bed
        z_surf, z_surf2 = z_surf2, z_surf
    end
    Rinv = Data.Array(R')
    @parallel set_phases!(ϕ,z_surf,z_bed,Rinv,xc_l[1],yc_l[1],zc_l[1],xc_ss[1],yc_ss[1],dx,dy,dz,dsx,dsy)
    update_halo!(ϕ)
    len_g = sum_g(ϕ.==fluid)
    # visu
    if do_save
        (me==0) && !ispath(out_path) && mkdir(out_path)
        Vn  = @zeros(nx-2,ny-2,nz-2)
        τII = @zeros(nx-2,ny-2,nz-2)
    end
    (me==0) && println(" done. Starting the real stuff 🚀")
    # time 
    ts = Float64[]; tt = 0.0; h5_names = String[]; isave = 1
    for it = 1:nt
        if (me==0) @printf("➡ it = %d\n", it) end
        T_o .= T
        # iteration loop
        err_V=2*ε_V; err_∇V=2*ε_∇V; err_T=2*ε_T; iter=0; err_evo1=[]; err_evo2=[]
        while !((err_V <= ε_V) && (err_∇V <= ε_∇V) && (err_T <= ε_T)) && (iter <= maxiter)
            @parallel compute_EII!(EII, Vx, Vy, Vz, ϕ, dx, dy, dz)
            @parallel (1:size(EII,2), 1:size(EII,3)) bc_x!(EII)
            @parallel (1:size(EII,1), 1:size(EII,3)) bc_y!(EII)
            @parallel (1:size(EII,1), 1:size(EII,2)) bc_z!(EII)
            @hide_communication b_width begin
                @parallel compute_P_τ_qT!(∇V, Pt, τxx, τyy, τzz, τxy, τxz, τyz, qTx, qTy, qTz, Vx, Vy, Vz, μs, ϕ, T, vpdτ_mech, Re_mech, r_mech, max_lxyz, χ, θr_dτ, dx, dy, dz)
                update_halo!(qTx,qTy,qTz,EII)
            end
            @hide_communication b_width begin
                @parallel compute_V_T_μ!(Vx, Vy, Vz, T, μs, Pt, τxx, τyy, τzz, τxy, τxz, τyz, EII, T_o, qTx, qTy, qTz, ϕ, μs0, ρgx, ρgy, ρgz, Ta, Q_R, T0, dt, npow, γ, vpdτ_mech, max_lxyz, Re_mech, dτ_ρ_heat, dx, dy, dz)
                @parallel (1:size(μs,2), 1:size(μs,3)) bc_x!(μs)
                @parallel (1:size(μs,1), 1:size(μs,3)) bc_y!(μs)
                @parallel (1:size(μs,1), 1:size(μs,2)) bc_z!(μs)
                update_halo!(Vx,Vy,Vz,μs)
            end
            iter += 1
            if iter % nchk == 0
                @parallel compute_Res!(Rx, Ry, Rz, RT, Pt, τxx, τyy, τzz, τxy, τxz, τyz, T, T_o, qTx, qTy, qTz, EII, μs, ϕ, ρgx, ρgy, ρgz, dt, dx, dy, dz)
                norm_Rx = norm_g(Rx)/psc*lz/sqrt(len_g)
                norm_Ry = norm_g(Ry)/psc*lz/sqrt(len_g)
                norm_Rz = norm_g(Rz)/psc*lz/sqrt(len_g)
                norm_∇V = norm_g(∇V)/vsc*lz/sqrt(len_g)
                # max_∇V  = max_g(abs.(∇V))/vsc*lz
                norm_T  = norm_g(RT)*tsc/ΔT/sqrt(len_g)
                err_V   = maximum([norm_Rx, norm_Ry, norm_Rz])
                err_∇V  = norm_∇V
                err_T   = norm_T
                any(isnan.([err_V,err_∇V,err_T])) && error("NaN")
                # push!(err_evo1, maximum([norm_Rx, norm_Ry, norm_∇V])); push!(err_evo2,iter/nx)
                if (me==0) @printf("# iters = %d, err_V = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_Rz=%1.3e], err_∇V = %1.3e err_T = %1.3e \n", iter, err_V, norm_Rx, norm_Ry, norm_Rz, err_∇V, err_T) end
                # GC.gc() # force garbage collection
            end
        end
        tt += dt
        if do_save && (it % nsave == 0)
            dim_g = (nx_g()-2, ny_g()-2, nz_g()-2)
            @parallel preprocess_visu!(Vn, τII, Vx, Vy, Vz, τxx, τyy, τzz, τxy, τxz, τyz)
            @parallel apply_mask!(Vn, τII, ϕ)
            out_h5 = joinpath(out_path,out_name)*"_$isave.h5"
            I = CartesianIndices(( (coords[1]*(nx-2) + 1):(coords[1]+1)*(nx-2),
                                   (coords[2]*(ny-2) + 1):(coords[2]+1)*(ny-2),
                                   (coords[3]*(nz-2) + 1):(coords[3]+1)*(nz-2) ))
            fields = Dict("ϕ"=>inn(ϕ),"Vn"=>Vn,"τII"=>τII,"Pr"=>inn(Pt),"EII"=>inn(EII),"T"=>inn(T),"μ"=>inn(μs))
            push!(ts,tt); push!(h5_names,out_name*"_$isave.h5")
            (me==0) && print("Saving HDF5 file...")
            write_h5(out_h5,fields,dim_g,I,comm_cart,info) # comm_cart,MPI.Info() are varargs to exclude if using non-parallel HDF5 lib
            # write_h5(out_h5,fields,dim_g,I) # comm_cart,MPI.Info() are varargs to exclude if using non-parallel HDF5 lib
            (me==0) && println(" done")
            # write XDMF
            if me==0
                print("Saving XDMF file...")
                write_xdmf(joinpath(out_path,out_name)*".xdmf3",h5_names,fields,(xc[2],yc[2],zc[2]),(dx,dy,dz),dim_g,ts)
                println(" done")
            end
            isave += 1
        end
    end
    finalize_global_grid()
    return
end

Stokes3D(load_elevation("../data/alps/data_Rhone.h5"))

# Stokes3D(generate_elevation(5.0,5.0,(0.0,1.0),0.0,0π,tan(-π/6),0.5,0.9))
