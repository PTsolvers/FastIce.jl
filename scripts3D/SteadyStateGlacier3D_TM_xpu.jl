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

@views inn(A) = A[2:end-1,2:end-1,2:end-1]
@views av(A)  = convert(eltype(A),0.5)*(A[1:end-1]+A[2:end])

include(joinpath(@__DIR__, "helpers3D_v5.jl"))
include(joinpath(@__DIR__, "data_io.jl"     ))

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

@parallel_indices (ix,iy,iz) function compute_EII!(EII, Vx, Vy, Vz, ϕ, dx, dy, dz)
    nfluid_xy = 0; nfluid_xz = 0; nfluid_yz = 0; 
    exy = 0.0; exz = 0.0; eyz = 0.0; exx = 0.0; eyy = 0.0; ezz = 0.0
    if ix <= size(EII,1)-2 && iy <= size(EII,2)-2 && iz <= size(EII,3)-2
        if ϕ[ix,iy,iz+1] == fluid && ϕ[ix+1,iy,iz+1] == fluid && ϕ[ix,iy+1,iz+1] == fluid && ϕ[ix+1,iy+1,iz+1] == fluid
            nfluid_xy += 1
            exy    += (Vx[ix+1,iy+1,iz+1] - Vx[ix+1,iy,iz+1])/dy + (Vy[ix+1,iy+1,iz+1] - Vy[ix,iy+1,iz+1])/dx
        end
        if ϕ[ix+1,iy,iz+1] == fluid && ϕ[ix+2,iy,iz+1] == fluid && ϕ[ix+1,iy+1,iz+1] == fluid && ϕ[ix+2,iy+1,iz+1] == fluid
            nfluid_xy += 1
            exy    += (Vx[ix+2,iy+1,iz+1] - Vx[ix+2,iy,iz+1])/dy + (Vy[ix+2,iy+1,iz+1] - Vy[ix+1,iy+1,iz+1])/dx
        end
        if ϕ[ix,iy+1,iz+1] == fluid && ϕ[ix+1,iy+1,iz+1] == fluid && ϕ[ix,iy+2,iz+1] == fluid && ϕ[ix+1,iy+2,iz+1] == fluid
            nfluid_xy += 1
            exy    += (Vx[ix+1,iy+2,iz+1] - Vx[ix+1,iy+1,iz+1])/dy + (Vy[ix+1,iy+2,iz+1] - Vy[ix,iy+2,iz+1])/dx
        end
        if ϕ[ix+1,iy+1,iz+1] == fluid && ϕ[ix+2,iy+1,iz+1] == fluid && ϕ[ix+1,iy+2,iz+1] == fluid && ϕ[ix+2,iy+2,iz+1] == fluid
            nfluid_xy += 1
            exy    += (Vx[ix+2,iy+2,iz+1] - Vx[ix+2,iy+1,iz+1])/dy + (Vy[ix+2,iy+2,iz+1] - Vy[ix+1,iy+2,iz+1])/dx
        end
        # ----------------------------------------------------------------------------------------------------
        if ϕ[ix,iy+1,iz] == fluid && ϕ[ix+1,iy+1,iz] == fluid && ϕ[ix,iy+1,iz+1] == fluid && ϕ[ix+1,iy+1,iz+1] == fluid
            nfluid_xz += 1
            exz    += (Vx[ix+1,iy+1,iz+1] - Vx[ix+1,iy+1,iz])/dz + (Vz[ix+1,iy+1,iz+1] - Vz[ix,iy+1,iz+1])/dx
        end
        if ϕ[ix+1,iy+1,iz] == fluid && ϕ[ix+2,iy+1,iz] == fluid && ϕ[ix+1,iy+1,iz+1] == fluid && ϕ[ix+2,iy+1,iz+1] == fluid
            nfluid_xz += 1
            exz    += (Vx[ix+2,iy+1,iz+1] - Vx[ix+2,iy+1,iz])/dz + (Vz[ix+2,iy+1,iz+1] - Vz[ix+1,iy+1,iz+1])/dx
        end
        if ϕ[ix,iy+1,iz+1] == fluid && ϕ[ix+1,iy+1,iz+1] == fluid && ϕ[ix,iy+1,iz+2] == fluid && ϕ[ix+1,iy+1,iz+2] == fluid
            nfluid_xz += 1
            exz    += (Vx[ix+1,iy+1,iz+2] - Vx[ix+1,iy+1,iz+1])/dz + (Vz[ix+1,iy+1,iz+2] - Vz[ix,iy+1,iz+2])/dx
        end
        if ϕ[ix+1,iy+1,iz+1] == fluid && ϕ[ix+2,iy+1,iz+1] == fluid && ϕ[ix+1,iy+1,iz+2] == fluid && ϕ[ix+2,iy+1,iz+2] == fluid
            nfluid_xz += 1
            exz    += (Vx[ix+2,iy+1,iz+2] - Vx[ix+2,iy+1,iz+1])/dz + (Vz[ix+2,iy+1,iz+2] - Vz[ix+1,iy+1,iz+2])/dx
        end
        # ----------------------------------------------------------------------------------------------------
        if ϕ[ix+1,iy,iz] == fluid && ϕ[ix+1,iy+1,iz] == fluid && ϕ[ix+1,iy,iz+1] == fluid && ϕ[ix+1,iy+1,iz+1] == fluid
            nfluid_yz += 1
            eyz    += (Vy[ix+1,iy+1,iz+1] - Vy[ix+1,iy+1,iz])/dz + (Vz[ix+1,iy+1,iz+1] - Vz[ix+1,iy,iz+1])/dy
        end
        if ϕ[ix+1,iy+1,iz] == fluid && ϕ[ix+1,iy+2,iz] == fluid && ϕ[ix+1,iy+1,iz+1] == fluid && ϕ[ix+1,iy+2,iz+1] == fluid
            nfluid_yz += 1
            eyz    += (Vy[ix+1,iy+2,iz+1] - Vy[ix+1,iy+2,iz])/dz + (Vz[ix+1,iy+2,iz+1] - Vz[ix+1,iy+1,iz+1])/dy
        end
        if ϕ[ix+1,iy,iz+1] == fluid && ϕ[ix+1,iy+1,iz+1] == fluid && ϕ[ix+1,iy,iz+2] == fluid && ϕ[ix+1,iy+1,iz+2] == fluid
            nfluid_yz += 1
            eyz    += (Vy[ix+1,iy+1,iz+2] - Vy[ix+1,iy+1,iz+1])/dz + (Vz[ix+1,iy+1,iz+2] - Vz[ix+1,iy,iz+2])/dy
        end
        if ϕ[ix+1,iy+1,iz+1] == fluid && ϕ[ix+1,iy+2,iz+1] == fluid && ϕ[ix+1,iy+1,iz+2] == fluid && ϕ[ix+1,iy+2,iz+2] == fluid
            nfluid_yz += 1
            eyz    += (Vy[ix+1,iy+2,iz+2] - Vy[ix+1,iy+2,iz+1])/dz + (Vz[ix+1,iy+2,iz+2] - Vz[ix+1,iy+1,iz+2])/dy
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

macro Gdτ()          esc(:(vpdτ_mech*Re_mech*@all(μs)/max_lxyz/(r+2.0)    )) end
macro Gdτ_av_xyi()   esc(:(vpdτ_mech*Re_mech*@av_xyi(μs)/max_lxyz/(r+2.0) )) end
macro Gdτ_av_xzi()   esc(:(vpdτ_mech*Re_mech*@av_xzi(μs)/max_lxyz/(r+2.0) )) end
macro Gdτ_av_yzi()   esc(:(vpdτ_mech*Re_mech*@av_yzi(μs)/max_lxyz/(r+2.0) )) end
macro μ_veτ()        esc(:(1.0/(1.0/@Gdτ()        + 1.0/@all(μs))         )) end
macro μ_veτ_av_xyi() esc(:(1.0/(1.0/@Gdτ_av_xyi() + 1.0/@av_xyi(μs))      )) end
macro μ_veτ_av_xzi() esc(:(1.0/(1.0/@Gdτ_av_xzi() + 1.0/@av_xzi(μs))      )) end
macro μ_veτ_av_yzi() esc(:(1.0/(1.0/@Gdτ_av_yzi() + 1.0/@av_yzi(μs))      )) end

@parallel function compute_P_τ_qT!(∇V, Pt, τxx, τyy, τzz, τxy, τxz, τyz, qTx, qTy, qTz, Vx, Vy, Vz, μs, ϕ, T, vpdτ_mech, Re_mech, r, max_lxyz, χ, θr_dτ, dx, dy, dz)
    @all(∇V)  = @fm(ϕ)*(@d_xa(Vx)/dx + @d_ya(Vy)/dy + @d_za(Vz)/dz)
    @all(Pt)  = @fm(ϕ)*(@all(Pt) - r*@Gdτ()*@all(∇V))    
    @all(τxx) = @fm(ϕ)*2.0*@μ_veτ()*(@d_xa(Vx)/dx + @all(τxx)/@Gdτ()/2.0)
    @all(τyy) = @fm(ϕ)*2.0*@μ_veτ()*(@d_ya(Vy)/dy + @all(τyy)/@Gdτ()/2.0)
    @all(τzz) = @fm(ϕ)*2.0*@μ_veτ()*(@d_za(Vz)/dz + @all(τzz)/@Gdτ()/2.0)
    @all(τxy) = @fmxy(ϕ)*2.0*@μ_veτ_av_xyi()*(0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx) + @all(τxy)/@Gdτ_av_xyi()/2.0)
    @all(τxz) = @fmxz(ϕ)*2.0*@μ_veτ_av_xzi()*(0.5*(@d_zi(Vx)/dz + @d_xi(Vz)/dx) + @all(τxz)/@Gdτ_av_xzi()/2.0)
    @all(τyz) = @fmyz(ϕ)*2.0*@μ_veτ_av_yzi()*(0.5*(@d_zi(Vy)/dz + @d_yi(Vz)/dy) + @all(τyz)/@Gdτ_av_yzi()/2.0)
    # thermo
    @inn_x(qTx) = (@inn_x(qTx) * θr_dτ - χ*@d_xa(T)/dx) / (θr_dτ + 1.0)
    @inn_y(qTy) = (@inn_y(qTy) * θr_dτ - χ*@d_ya(T)/dy) / (θr_dτ + 1.0)
    @inn_z(qTz) = (@inn_z(qTz) * θr_dτ - χ*@d_za(T)/dz) / (θr_dτ + 1.0)
    return
end

macro sm_xi(A) esc(:( !(($A[$ix,$iyi,$izi] == solid) || ($A[$ix+1,$iyi,$izi] == solid)) )) end
macro sm_yi(A) esc(:( !(($A[$ixi,$iy,$izi] == solid) || ($A[$ixi,$iy+1,$izi] == solid)) )) end
macro sm_zi(A) esc(:( !(($A[$ixi,$iyi,$iz] == solid) || ($A[$ixi,$iyi,$iz+1] == solid)) )) end

macro fm_xi(A) esc(:( ($A[$ix,$iyi,$izi] == fluid) && ($A[$ix+1,$iyi,$izi] == fluid) )) end
macro fm_yi(A) esc(:( ($A[$ixi,$iy,$izi] == fluid) && ($A[$ixi,$iy+1,$izi] == fluid) )) end
macro fm_zi(A) esc(:( ($A[$ixi,$iyi,$iz] == fluid) && ($A[$ixi,$iyi,$iz+1] == fluid) )) end

macro dτ_ρ_mech_ax() esc(:( vpdτ_mech*max_lxyz/Re_mech/@av_xi(μs) )) end
macro dτ_ρ_mech_ay() esc(:( vpdτ_mech*max_lxyz/Re_mech/@av_yi(μs) )) end
macro dτ_ρ_mech_az() esc(:( vpdτ_mech*max_lxyz/Re_mech/@av_zi(μs) )) end

@parallel function compute_V_T_μ!(Vx, Vy, Vz, T, μs, Pt, τxx, τyy, τzz, τxy, τxz, τyz, EII, T_o, qTx, qTy, qTz, ϕ, μs0, ρgx, ρgy, ρgz, Ta, Q_R, T0, dt, npow, γ, vpdτ_mech, max_lxyz, Re_mech, dτ_ρ_heat, dx, dy, dz)
    @inn(Vx) = @sm_xi(ϕ)*( @inn(Vx) + @dτ_ρ_mech_ax()*(@d_xi(τxx)/dx + @d_ya(τxy)/dy + @d_za(τxz)/dz - @d_xi(Pt)/dx - @fm_xi(ϕ)*ρgx) )
    @inn(Vy) = @sm_yi(ϕ)*( @inn(Vy) + @dτ_ρ_mech_ay()*(@d_yi(τyy)/dy + @d_xa(τxy)/dx + @d_za(τyz)/dz - @d_yi(Pt)/dy - @fm_yi(ϕ)*ρgy) )
    @inn(Vz) = @sm_zi(ϕ)*( @inn(Vz) + @dτ_ρ_mech_az()*(@d_zi(τzz)/dy + @d_xa(τxz)/dx + @d_ya(τyz)/dy - @d_zi(Pt)/dz - @fm_zi(ϕ)*ρgz) )
    # thermo
    @all(T)  = (@all(T) + dτ_ρ_heat*(@all(T_o)/dt - @d_xa(qTx)/dx - @d_ya(qTy)/dy - @d_za(qTz)/dz + 2.0*@all(μs)*@all(EII)))/(1.0 + dτ_ρ_heat/dt)
    @all(μs) = (1.0-γ)*@all(μs) + γ*(( @all(EII)^(1.0/npow-1.0) * exp(-Q_R*(1.0 - T0/@all(T))) )^(-1) + 1.0/μs0)^(-1)
    return
end

@parallel function compute_Res!(Rx, Ry, Rz, RT, Pt, τxx, τyy, τzz, τxy, τxz, τyz, T, T_o, qTx, qTy, qTz, EII, μs, ϕ, ρgx, ρgy, ρgz, dt, dx, dy, dz)
    @all(Rx) = @sm_xi(ϕ)*(@d_xi(τxx)/dx + @d_ya(τxy)/dy + @d_za(τxz)/dz - @d_xi(Pt)/dx - @fm_xi(ϕ)*ρgx)
    @all(Ry) = @sm_yi(ϕ)*(@d_yi(τyy)/dy + @d_xa(τxy)/dx + @d_za(τyz)/dz - @d_yi(Pt)/dy - @fm_yi(ϕ)*ρgy)
    @all(Rz) = @sm_zi(ϕ)*(@d_zi(τzz)/dy + @d_xa(τxz)/dx + @d_ya(τyz)/dy - @d_zi(Pt)/dz - @fm_zi(ϕ)*ρgz)
    # thermo
    @all(RT) = -(@all(T) - @all(T_o))/dt - (@d_xa(qTx)/dx + @d_ya(qTy)/dy + @d_za(qTz)/dz) + 2.0*@all(μs)*@all(EII)
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

@parallel_indices (ix,iy,iz) function set_phases!(ϕ,zsurf,zbed,R,ox,oy,oz,osx,osy,dx,dy,dz,dsx,dsy,cx,cy,cz)
    if checkbounds(Bool,ϕ,ix,iy,iz)
        ixg,iyg,izg = ix + cx*(size(ϕ,1)-2), iy + cy*(size(ϕ,2)-2), iz + cz*(size(ϕ,3)-2)
        xc,yc,zc    = ox + (ixg-1)*dx, oy + (iyg-1)*dy, oz + (izg-1)*dz
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

function apply_bc!(A)
    @parallel (1:size(A,2), 1:size(A,3)) bc_x!(A)
    @parallel (1:size(A,1), 1:size(A,3)) bc_y!(A)
    @parallel (1:size(A,1), 1:size(A,2)) bc_z!(A)
    return
end

@views function Stokes3D(dem)
    # inputs
    # nx,ny,nz  = 511,511,383      # local resolution
    nx,ny,nz  = 127,127,95       # local resolution
    # nx,ny,nz  = 63,63,47         # local resolution
    nt        = 500                # number of timesteps
    dim       = (2,2,2)          # MPI dims
    ns        = 2                # number of oversampling per cell
    out_path  = "../out_visu"
    out_name  = "results3D_TM"
    nsave     = 10
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
    # physics
    ## dimensionally independent
    μs0       = 1.0               # matrix viscosity [Pa*s]
    ρg0       = 1.0               # gravity          [Pa/m]
    ΔT        = 1.0              # temperature difference between ice and atmosphere [K]
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
    maxiter   = 100*nz_g()    # maximum number of pseudo-transient iterations
    nchk      = 6*nz_g()      # error checking frequency
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
    ϕx        = @zeros(nx-1,ny-2,nz-2)
    ϕy        = @zeros(nx-2,ny-1,nz-2)
    ϕz        = @zeros(nx-2,ny-2,nz-1)
    Vx        = @zeros(nx+1,ny  ,nz  )
    Vy        = @zeros(nx  ,ny+1,nz  )
    Vz        = @zeros(nx  ,ny  ,nz+1)
    EII       = @zeros(nx  ,ny  ,nz  )
    T_o       = @zeros(nx  ,ny  ,nz  )
    qTx       = @zeros(nx+1,ny  ,nz  )
    qTy       = @zeros(nx  ,ny+1,nz  )
    qTz       = @zeros(nx  ,ny  ,nz+1)
    μs        = 1e1μs0 .* @ones(nx,ny,nz)
    T         =    T0  .* @ones(nx,ny,nz)
    # set phases
    if (me==0) print("Set phases (0-air, 1-ice, 2-bedrock)...") end
    Rinv         = Data.Array(R')
    nr_box       = dem.domain
    xc_ss,yc_ss  = LinRange(nr_box.xmin,nr_box.xmax,ns*length(xc)),LinRange(nr_box.ymin,nr_box.ymax,ns*length(yc)) # supersampled grid
    dsx,dsy      = xc_ss[2] - xc_ss[1], yc_ss[2] - yc_ss[1]
    z_bed,z_surf = Data.Array.(evaluate(dem, xc_ss, yc_ss))
    ϕ            = air.*@ones(nx,ny,nz)
    @parallel set_phases!(ϕ,z_surf,z_bed,Rinv,xc[1],yc[1],zc[1],xc_ss[1],yc_ss[1],dx,dy,dz,dsx,dsy,coords...)
    @parallel init_ϕi!(ϕ, ϕx, ϕy, ϕz)
    len_g = sum_g(ϕ.==fluid)
    # visu
    if do_save
        (me==0) && !ispath(out_path) && mkdir(out_path)
        Vn   = @zeros(nx-2,ny-2,nz-2)
        τII  = @zeros(nx-2,ny-2,nz-2)
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
            apply_bc!(EII)
            @hide_communication b_width begin
                @parallel compute_P_τ_qT!(∇V, Pt, τxx, τyy, τzz, τxy, τxz, τyz, qTx, qTy, qTz, Vx, Vy, Vz, μs, ϕ, T, vpdτ_mech, Re_mech, r_mech, max_lxyz, χ, θr_dτ, dx, dy, dz)
                update_halo!(qTx,qTy,qTz,EII)
            end
            @hide_communication b_width begin
                @parallel compute_V_T_μ!(Vx, Vy, Vz, T, μs, Pt, τxx, τyy, τzz, τxy, τxz, τyz, EII, T_o, qTx, qTy, qTz, ϕ, μs0, ρgx, ρgy, ρgz, Ta, Q_R, T0, dt, npow, γ, vpdτ_mech, max_lxyz, Re_mech, dτ_ρ_heat, dx, dy, dz)
                apply_bc!(μs)
                update_halo!(Vx,Vy,Vz,μs)
            end
            iter += 1
            if iter % nchk == 0
                @parallel compute_Res!(Rx, Ry, Rz, RT, Pt, τxx, τyy, τzz, τxy, τxz, τyz, T, T_o, qTx, qTy, qTz, EII, μs, ϕ, ρgx, ρgy, ρgz, dt, dx, dy, dz)
                norm_Rx = norm_g(Rx)/psc*lz/sqrt(len_g)
                norm_Ry = norm_g(Ry)/psc*lz/sqrt(len_g)
                norm_Rz = norm_g(Rz)/psc*lz/sqrt(len_g)
                norm_∇V = norm_g(∇V)/vsc*lz/sqrt(len_g)
                norm_T  = norm_g(RT)*tsc/ΔT/sqrt(len_g)
                err_V   = maximum([norm_Rx, norm_Ry, norm_Rz])
                err_∇V  = norm_∇V
                err_T   = norm_T
                # push!(err_evo1, maximum([norm_Rx, norm_Ry, norm_∇V])); push!(err_evo2,iter/nx)
                if (me==0) @printf("# iters = %d, err_V = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_Rz=%1.3e], err_∇V = %1.3e err_T = %1.3e \n", iter, err_V, norm_Rx, norm_Ry, norm_Rz, err_∇V, err_T) end
                GC.gc() # force garbage collection
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
            push!(ts,tt); push!(h5_names,out_h5)
            (me==0) && print("Saving HDF5 file...")
            write_h5(out_h5,fields,dim_g,I,comm_cart,info) # comm_cart,MPI.Info() are varargs
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

# Stokes3D(load_elevation("../data/alps/data_Rhone.h5"))
Stokes3D(generate_elevation(2.0,2.0,(-0.25,0.85),1/25,10π,tan(-π/12),0.1,0.9))
