using AMDGPU
using GeometryBasics,LinearAlgebra,ElasticArrays,Printf,Random
using ImplicitGlobalGrid
import MPI
# using GLMakie

include("FD_helpers.jl")
const DAT = Float64

@views amean1(A) = 0.5.*(A[1:end-1] .+ A[2:end])
@views ameanx(A) = 0.5.*(A[1:end-1,:,:] .+ A[2:end,:,:])
@views ameany(A) = 0.5.*(A[:,1:end-1,:] .+ A[:,2:end,:])
@views ameanz(A) = 0.5.*(A[:,:,1:end-1] .+ A[:,:,2:end])

maximum_g(A) = (max_l = maximum(A); MPI.Allreduce(max_l,MPI.MAX,MPI.COMM_WORLD))
# maximum_g(A) = (max_l = maximum(A))

function save_array(Aname,A)
    fname = string(Aname,".bin")
    out = open(fname,"w"); write(out,A); close(out)
end

function update_iter_params!(ητ,η,b_width,istep)
    @get_thread_idx()
    # CommOverlap
    nx,ny,nz = size(ητ)
    if ( istep==1 && ( ix> b_width[1] && ix< nx-b_width[1] && iy> b_width[2] && iy< ny-b_width[2] && iz> b_width[3] && iz< nz-b_width[3] ) ); @goto early_exit end
    if ( istep==2 && ( ix<=b_width[1] || ix>=nx-b_width[1] || iy<=b_width[2] || iy>=ny-b_width[2] || iz<=b_width[3] || iz>=nz-b_width[3] ) ); @goto early_exit end
    if (ix<=size(ητ,1)-2 && iy<=size(ητ,2)-2 && iz<=size(ητ,3)-2) @inn(ητ) = @maxloc(η) end
    @label early_exit
    return
end

function bc_z!(A)
    @get_thread_idx()
    nx,ny,nz=size(A)
    if (ix<=nx && iy<=ny && iz==1 ) A[ix,iy,iz] = A[ix,iy,iz+1] end
    if (ix<=nx && iy<=ny && iz==nz) A[ix,iy,iz] = A[ix,iy,iz-1] end
    return
end

function bc_y!(A)
    @get_thread_idx()
    nx,ny,nz=size(A)
    if (ix<=nx && iy==1  && iz<=nz) A[ix,iy,iz] = A[ix,iy+1,iz] end
    if (ix<=nx && iy==ny && iz<=nz) A[ix,iy,iz] = A[ix,iy-1,iz] end
    return
end

function bc_x!(A)
    @get_thread_idx()
    nx,ny,nz=size(A)
    if (ix==1  && iy<=ny && iz<=nz) A[ix,iy,iz] = A[ix+1,iy,iz] end
    if (ix==nx && iy<=ny && iz<=nz) A[ix,iy,iz] = A[ix-1,iy,iz] end
    return
end

function update_P_ε!(Pr,dPr,εxx,εyy,εzz,εxy,εxz,εyz,Vx,Vy,Vz,∇V,η,r,θ_dτ,dx,dy,dz)
    @get_thread_idx()
    if (ix<=size(∇V,1) && iy<=size(∇V,2) && iz<=size(∇V,3))
        @all(∇V)  = @d_xa(Vx)/dx + @d_ya(Vy)/dy + @d_za(Vz)/dz
        @all(dPr) = -@all(∇V)
        @all(Pr)  = @all(Pr) + @all(dPr)*@all(η)*r/θ_dτ
        @all(εxx) = @d_xa(Vx)/dx - @all(∇V)/3.0
        @all(εyy) = @d_ya(Vy)/dy - @all(∇V)/3.0
        @all(εzz) = @d_za(Vz)/dz - @all(∇V)/3.0
    end
    if (ix<=size(εxy,1) && iy<=size(εxy,2) && iz<=size(εxy,3)) @all(εxy) = 0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx) end
    if (ix<=size(εxz,1) && iy<=size(εxz,2) && iz<=size(εxz,3)) @all(εxz) = 0.5*(@d_zi(Vx)/dz + @d_xi(Vz)/dx) end
    if (ix<=size(εyz,1) && iy<=size(εyz,2) && iz<=size(εyz,3)) @all(εyz) = 0.5*(@d_zi(Vy)/dz + @d_yi(Vz)/dy) end
    return
end

function update_τ!(τxx,τyy,τzz,τxy,τxz,τyz,εxx,εyy,εzz,εxy,εxz,εyz,εII,η,dτ_r)
    @get_thread_idx()
    if (ix<=size(τxx,1) && iy<=size(τxx,2) && iz<=size(τxx,3))
        @all(τxx) = @all(τxx) + (-@all(τxx) + 2.0*@all(η)*@all(εxx))*dτ_r
        @all(τyy) = @all(τyy) + (-@all(τyy) + 2.0*@all(η)*@all(εyy))*dτ_r
        @all(τzz) = @all(τzz) + (-@all(τzz) + 2.0*@all(η)*@all(εzz))*dτ_r
    end
    if (ix<=size(τxy,1) && iy<=size(τxy,2) && iz<=size(τxy,3)) @all(τxy) = @all(τxy) + (-@all(τxy) + 2.0*@av_xyi(η)*@all(εxy))*dτ_r end
    if (ix<=size(τxz,1) && iy<=size(τxz,2) && iz<=size(τxz,3)) @all(τxz) = @all(τxz) + (-@all(τxz) + 2.0*@av_xzi(η)*@all(εxz))*dτ_r end
    if (ix<=size(τyz,1) && iy<=size(τyz,2) && iz<=size(τyz,3)) @all(τyz) = @all(τyz) + (-@all(τyz) + 2.0*@av_yzi(η)*@all(εyz))*dτ_r end
    if (ix<=size(εII,1) && iy<=size(εII,2) && iz<=size(εII,3)) @all(εII) = sqrt(0.5*(@inn(εxx)^2 + @inn(εyy)^2 + @inn(εzz)^2) + @av_xya(εxy)^2 + @av_xza(εxz)^2 + @av_yza(εyz)^2) end
    return
end

function update_qU!(qUx,qUy,qUz,T,λ,dx,dy,dz)
    @get_thread_idx()
    if (ix<=size(qUx,1) && iy<=size(qUx,2) && iz<=size(qUx,3)) @all(qUx) = -λ*@d_xi(T)/dx end
    if (ix<=size(qUy,1) && iy<=size(qUy,2) && iz<=size(qUy,3)) @all(qUy) = -λ*@d_yi(T)/dy end
    if (ix<=size(qUz,1) && iy<=size(qUz,2) && iz<=size(qUz,3)) @all(qUz) = -λ*@d_zi(T)/dz end
    return
end

function update_U!(U,qUx,qUy,qUz,η,εII,dt,dx,dy,dz,b_width,istep)
    @get_thread_idx()
    # CommOverlap
    nx,ny,nz = size(U)
    if ( istep==1 && ( ix> b_width[1] && ix< nx-b_width[1] && iy> b_width[2] && iy< ny-b_width[2] && iz> b_width[3] && iz< nz-b_width[3] ) ); @goto early_exit end
    if ( istep==2 && ( ix<=b_width[1] || ix>=nx-b_width[1] || iy<=b_width[2] || iy>=ny-b_width[2] || iz<=b_width[3] || iz>=nz-b_width[3] ) ); @goto early_exit end
    if (ix<=size(U,1)-2 && iy<=size(U,2)-2 && iz<=size(U,3)-2)
        @inn(U) = @inn(U) + dt*(-(@d_xa(qUx)/dx + @d_ya(qUy)/dy + @d_za(qUz)/dz) + 2.0*@inn(η)*@all(εII)^2)
    end
    @label early_exit
    return
end

function compute_T!(T,U,ρCp)
    @get_thread_idx()
    if (ix<=size(T,1) && iy<=size(T,2) && iz<=size(T,3)) @all(T) = @all(U)/ρCp end
    return
end

function update_velocities!(Vx,Vy,Vz,Pr,τxx,τyy,τzz,τxy,τxz,τyz,ητ,ρgx,ρgy,ρgz,nudτ,dx,dy,dz,b_width,istep)
    @get_thread_idx()
    # CommOverlap
    nx,ny,nz = size(Pr)
    if ( istep==1 && ( ix> b_width[1] && ix< nx-b_width[1] && iy> b_width[2] && iy< ny-b_width[2] && iz> b_width[3] && iz< nz-b_width[3] ) ); @goto early_exit end
    if ( istep==2 && ( ix<=b_width[1] || ix>=nx-b_width[1] || iy<=b_width[2] || iy>=ny-b_width[2] || iz<=b_width[3] || iz>=nz-b_width[3] ) ); @goto early_exit end
    if (ix<=size(Vx,1)-2 && iy<=size(Vx,2)-2 && iz<=size(Vx,3)-2) @inn(Vx) = @inn(Vx) + (-@d_xi(Pr)/dx + @d_xi(τxx)/dx + @d_ya(τxy)/dy + @d_za(τxz)/dz - @all(ρgx))*nudτ/@av_xi(ητ) end
    if (ix<=size(Vy,1)-2 && iy<=size(Vy,2)-2 && iz<=size(Vy,3)-2) @inn(Vy) = @inn(Vy) + (-@d_yi(Pr)/dy + @d_yi(τyy)/dy + @d_xa(τxy)/dx + @d_za(τyz)/dz - @all(ρgy))*nudτ/@av_yi(ητ) end
    if (ix<=size(Vz,1)-2 && iy<=size(Vz,2)-2 && iz<=size(Vz,3)-2) @inn(Vz) = @inn(Vz) + (-@d_zi(Pr)/dz + @d_zi(τzz)/dz + @d_xa(τxz)/dx + @d_ya(τyz)/dy - @all(ρgz))*nudτ/@av_zi(ητ) end
    @label early_exit
    return
end

function bc_Vxy!(Vxy)
    @get_thread_idx()
    nx,ny,nz=size(Vxy)
    if (ix<=nx && iy<=ny && iz==1) Vxy[ix,iy,iz] = -Vxy[ix,iy,iz+1] end
    return
end

function bc_Vx!(Vx,η,Pr,ph_ice,ph_bed,dx)
    @get_thread_idx()
    nx,ny,nz=size(Pr); xE = ix+1
    if (ix==1 && iy<=ny && iz<=nz)
        t_air        = 1.0 - 0.5*(ph_ice[ix,iy,iz] + ph_ice[ix+1,iy,iz]) - 0.5*(ph_bed[ix,iy,iz] + ph_bed[ix+1,iy,iz])
        Vx[ix,iy,iz] = t_air*(Vx[ix+1,iy,iz] - 0.5*dx/η[ix,iy,iz]*(Pr[ix+1,iy,iz] + Pr[ix,iy,iz] - 2.0*η[ix+1,iy,iz]*(Vx[ix+2,iy,iz] - Vx[ix+1,iy,iz])/dx)) - (1.0-t_air)*Vx[ix+2,iy,iz]
    end
    if (ix==nx && iy<=ny && iz<=nz)
        t_air        = 1.0 - 0.5*(ph_ice[ix-1,iy,iz] + ph_ice[ix,iy,iz]) - 0.5*(ph_bed[ix-1,iy,iz] + ph_bed[ix,iy,iz])
        Vx[xE,iy,iz] = t_air*(Vx[xE-1,iy,iz] + 0.5*dx/η[ix,iy,iz]*(Pr[ix-1,iy,iz] + Pr[ix,iy,iz] - 2.0*η[ix-1,iy,iz]*(Vx[xE-1,iy,iz] - Vx[xE-2,iy,iz])/dx)) - (1.0-t_air)*Vx[xE-2,iy,iz]
    end
    return
end

function bc_Vy!(Vy,η,Pr,ph_ice,ph_bed,dy)
    @get_thread_idx()
    nx,ny,nz=size(Pr); yE = iy+1
    if (ix<=nx && iy==1 && iz<=nz)
        t_air        = 1.0 - 0.5*(ph_ice[ix,iy,iz] + ph_ice[ix,iy+1,iz]) - 0.5*(ph_bed[ix,iy,iz] + ph_bed[ix,iy+1,iz])
        Vy[ix,iy,iz] = t_air*(Vy[ix,iy+1,iz] - 0.5*dy/η[ix,iy,iz]*(Pr[ix,iy+1,iz] + Pr[ix,iy,iz] - 2.0*η[ix,iy+1,iz]*(Vy[ix,iy+2,iz] - Vy[ix,iy+1,iz])/dy)) - (1.0-t_air)*Vy[ix,iy+2,iz]
    end
    if (ix<=nx && iy==ny && iz<=nz)
        t_air        = 1.0 - 0.5*(ph_ice[ix,iy-1,iz] + ph_ice[ix,iy,iz]) - 0.5*(ph_bed[ix,iy-1,iz] + ph_bed[ix,iy,iz])
        Vy[ix,yE,iz] = t_air*(Vy[ix,yE-1,iz] + 0.5*dy/η[ix,iy,iz]*(Pr[ix,iy-1,iz] + Pr[ix,iy,iz] - 2.0*η[ix,iy-1,iz]*(Vy[ix,yE-1,iz] - Vy[ix,yE-2,iz])/dy)) - (1.0-t_air)*Vy[ix,yE-2,iz]
    end
    return
end

function bc_Vz!(Vz,η,Pr,dz)
    @get_thread_idx()
    nx,ny,nz=size(Pr); zE = iz+1
    if (ix<=nx && iy<=ny && iz==nz)
        Vz[ix,iy,zE] = Vz[ix,iy,zE-1] + 0.5*dz/η[ix,iy,iz]*(Pr[ix,iy,iz-1] + Pr[ix,iy,iz] - 2.0*η[ix,iy,iz-1]*(Vz[ix,iy,zE-1] - Vz[ix,iy,zE-2])/dz)
    end
    return
end

function compute_residuals!(r_Vx,r_Vy,r_Vz,Pr,τxx,τyy,τzz,τxy,τxz,τyz,ρgx,ρgy,ρgz,dx,dy,dz)
    @get_thread_idx()
    if (ix<=size(r_Vx,1) && iy<=size(r_Vx,2) && iz<=size(r_Vx,3)) @all(r_Vx) = -@d_xi(Pr)/dx + @d_xi(τxx)/dx + @d_ya(τxy)/dy + @d_za(τxz)/dz - @all(ρgx) end
    if (ix<=size(r_Vy,1) && iy<=size(r_Vy,2) && iz<=size(r_Vy,3)) @all(r_Vy) = -@d_yi(Pr)/dy + @d_yi(τyy)/dy + @d_xa(τxy)/dx + @d_za(τyz)/dz - @all(ρgy) end
    if (ix<=size(r_Vz,1) && iy<=size(r_Vz,2) && iz<=size(r_Vz,3)) @all(r_Vz) = -@d_zi(Pr)/dz + @d_zi(τzz)/dz + @d_xa(τxz)/dx + @d_ya(τyz)/dy - @all(ρgz) end
    return
end

function compute_η!(η,T,Q_R,η0_air,η0_ice,η0_bed,ph_ice,ph_bed)
    @get_thread_idx()
    if (ix<=size(η,1) && iy<=size(η,2) && iz<=size(η,3))
        t_ice   = @all(ph_ice)
        t_bed   = @all(ph_bed)
        t_air   = 1.0 - t_ice - t_bed
        η_ice   = η0_ice*exp(Q_R/@all(T))
        η_bed   = η0_bed
        η_air   = η0_air
        @all(η) = t_ice*η_ice + t_bed*η_bed + t_air*η_air
    end
    return
end

@inline function sd_round_box(p,b,r)
  q = abs.(p) .- b
  return norm(max.(q,0.0)) + min(max(q[1],max(q[2],q[3])),0.0) - r
end

function compute_phase!(ρgz_c,ph_ice,ph_bed,xc,yc,zc,r_box,w_box,r_rnd,z_bed,δ_sd,ρg0_air,ρg0_ice,ρg0_bed)
    @get_thread_idx()
    if (ix<=size(ρgz_c,1) && iy<=size(ρgz_c,2) && iz<=size(ρgz_c,3))
        nor    = normalize(Vec3(1.0,1.0,30.0))
        P      = Point(xc[ix],yc[iy],zc[iz])
        sd_bed = dot(nor,P-Point(0,0,z_bed))
        sd_ice = sd_round_box(P-r_box,w_box,r_rnd)
        sd_ice = max(sd_ice,-sd_bed)
        t_ice  = 0.5*(tanh(-sd_ice/δ_sd) + 1.0)
        t_bed  = 0.5*(tanh(-sd_bed/δ_sd) + 1.0)
        t_air  = 1.0 - t_ice - t_bed
        @all(ρgz_c)  = t_ice*ρg0_ice + t_bed*ρg0_bed + t_air*ρg0_air
        @all(ph_ice) = t_ice
        @all(ph_bed) = t_bed
    end
    return
end

function init_U!(U,T,ρCp,T0)
    @get_thread_idx()
    if (ix<=size(T,1) && iy<=size(T,2) && iz<=size(T,3))
        @all(T) = T0
        @all(U) = ρCp*@all(T)
    end
    return
end

@views function main(;do_save=false,outdir="out_mpi_$(randstring(3))")
    # physics
    lx,ly,lz   = 40.0,40.0,10.0
    η0         = (ice = 1.0  ,bed = 1e2  ,air = 1e-6 )
    ρg0        = (ice = 1.0  ,bed = 1.0  ,air = 0.0  )
    λ          = (ice = 1.0  ,bed = 1.0  ,air = 1.0  )
    ρCp        = (ice = 1.0  ,bed = 1.0  ,air = 1.0  )
    T0         = (ice = 253.0,bed = 253.0,air = 253.0)
    Q_R        = 10.0
    r_box      = Vec(0.0lx,0.0ly,0.1lz)
    w_box      = Vec(0.5lx-0.5lz,0.5ly-0.5lz,0.5lz)
    r_rnd      = 0.25lz
    z_bed      = 0.15lz
    # numerics
    nz         = 2*32-1
    nx         = ceil(Int,(nz+1)*lx/lz)-1
    ny         = ceil(Int,(nz+1)*ly/lz)-1
    dim        = (0,0,0)
    me,dims,nprocs,coords,comm_cart = init_global_grid(nx,ny,nz;dimx=dim[1],dimy=dim[2],dimz=dim[3])
    # me,dims,nprocs=0,(1,1,1),1
    ϵtol       = (1e-6,1e-6,1e-6,1e-6)
    maxiter    = 100min(nx_g(),ny_g(),nz_g())
    ncheck     = ceil(Int,5min(nx_g(),ny_g(),nz_g()))
    # maxiter    = 100min(nx,ny,nz)
    # ncheck     = ceil(Int,5min(nx,ny,nz))
    r          = 0.6
    re_mech    = 5.2π
    nt         = 10
    b_width    = (32,2,2)
    threads    = (128,2,1)
    grid       = (nx+1,ny+1,nz+1)
    me==0 && println("Process $me selecting device $(AMDGPU.default_device_id())")
    me==0 && println("Local problem size: nx=$nx, ny=$ny, nz=$nz")
    me==0 && println("ROCm grid=$grid, threads=$threads")
    # preprocessing
    # dx,dy,dz   = lx/nx,ly/ny,lz/nz
    # xv,yv,zv   = LinRange(-lx/2,lx/2,nx+1),LinRange(-ly/2,ly/2,ny+1),LinRange(0,lz,nz+1)
    # xc,yc,zc   = amean1(xv),amean1(yv),amean1(zv)
    xc,yc,zc   = zeros(nx,1,1),zeros(1,ny,1),zeros(1,1,nz)
    dx,dy,dz   = lx/nx_g(),ly/ny_g(),lz/nz_g()
    xc         = ROCArray([(x_g(ix,dx,xc)+dx/2-lx/2) for ix=1:size(xc,1)])
    yc         = ROCArray([(y_g(iy,dy,yc)+dy/2-ly/2) for iy=1:size(yc,2)])
    zc         = ROCArray([(z_g(iz,dz,zc)+dz/2     ) for iz=1:size(zc,3)])
    lτ         = min(lx,ly,lz)
    vdτ        = min(dx,dy,dz)/sqrt(3.1)
    θ_dτ       = lτ*(r+4/3)/(re_mech*vdτ)
    nudτ       = vdτ*lτ/re_mech
    dτ_r       = 1.0/(θ_dτ + 1.0)
    δ_sd       = 1.0*max(dx,dy,dz)
    dt         = min(dx,dy,dz)^2/(λ.ice/ρCp.ice)/6.1
    # array allocation
    Vx         = ROCArray(zeros(DAT,nx+1,ny  ,nz  ))
    Vy         = ROCArray(zeros(DAT,nx  ,ny+1,nz  ))
    Vz         = ROCArray(zeros(DAT,nx  ,ny  ,nz+1))
    Pr         = ROCArray(zeros(DAT,nx  ,ny  ,nz  ))
    ∇V         = ROCArray(zeros(DAT,nx  ,ny  ,nz  ))
    τxx        = ROCArray(zeros(DAT,nx  ,ny  ,nz  ))
    τyy        = ROCArray(zeros(DAT,nx  ,ny  ,nz  ))
    τzz        = ROCArray(zeros(DAT,nx  ,ny  ,nz  ))
    τxy        = ROCArray(zeros(DAT,nx-1,ny-1,nz-2))
    τxz        = ROCArray(zeros(DAT,nx-1,ny-2,nz-1))
    τyz        = ROCArray(zeros(DAT,nx-2,ny-1,nz-1))
    εxx        = ROCArray(zeros(DAT,nx  ,ny  ,nz  ))
    εyy        = ROCArray(zeros(DAT,nx  ,ny  ,nz  ))
    εzz        = ROCArray(zeros(DAT,nx  ,ny  ,nz  ))
    εxy        = ROCArray(zeros(DAT,nx-1,ny-1,nz-2))
    εxz        = ROCArray(zeros(DAT,nx-1,ny-2,nz-1))
    εyz        = ROCArray(zeros(DAT,nx-2,ny-1,nz-1))
    εII        = ROCArray(zeros(DAT,nx-2,ny-2,nz-2))
    η          = ROCArray(zeros(DAT,nx  ,ny  ,nz  ))
    Vmag       = ROCArray(zeros(DAT,nx  ,ny  ,nz  ))
    dPr        = ROCArray(zeros(DAT,nx  ,ny  ,nz  ))
    r_Vx       = ROCArray(zeros(DAT,nx-1,ny-2,nz-2))
    r_Vy       = ROCArray(zeros(DAT,nx-2,ny-1,nz-2))
    r_Vz       = ROCArray(zeros(DAT,nx-2,ny-2,nz-1))
    ρgz_c      = ROCArray(zeros(DAT,nx  ,ny  ,nz  ))
    ρgx        = ROCArray(zeros(DAT,nx-1,ny-2,nz-2))
    ρgy        = ROCArray(zeros(DAT,nx-2,ny-1,nz-2))
    ρgz        = ROCArray(zeros(DAT,nx-2,ny-2,nz-1))
    ph_ice     = ROCArray(zeros(DAT,nx  ,ny  ,nz  ))
    ph_bed     = ROCArray(zeros(DAT,nx  ,ny  ,nz  ))
    ητ         = ROCArray(zeros(DAT,nx  ,ny  ,nz  ))
    U          = ROCArray(zeros(DAT,nx  ,ny  ,nz  ))
    T          = ROCArray(zeros(DAT,nx  ,ny  ,nz  ))
    qUx        = ROCArray(zeros(DAT,nx-1,ny-2,nz-2))
    qUy        = ROCArray(zeros(DAT,nx-2,ny-1,nz-2))
    qUz        = ROCArray(zeros(DAT,nx-2,ny-2,nz-1))
    # AMDGPU specific
    nkern  = 25
    nkern2 = 3
    nstep  = 2 # 1 normal or 2 if comm overlap
    # rocqueues = Vector{AMDGPU.ROCQueue}(undef,nstep) # from IGG
    # for is = 1:nstep
    #     rocqueues[is] = is == 1 ? ROCQueue(AMDGPU.default_device(); priority=:high) : ROCQueue(AMDGPU.default_device())
    # end
    sig  = Array{AMDGPU.ROCSignal}(undef,nkern)
    sig2 = Array{AMDGPU.ROCSignal}(undef,nkern2,nstep)
    for ik=1:nkern
        sig[ik] = ROCSignal()
    end
    for ik=1:nkern2, is=1:nstep
        sig2[ik,is] = ROCSignal()
    end
    if do_save
        ENV["GKSwstype"]="nul"
        nx_v,ny_v,nz_v = (nx-2)*dims[1],(ny-2)*dims[2],(nz-2)*dims[3]
        if (nx_v*ny_v*nz_v*sizeof(DAT) > 0.8*Sys.free_memory()) error("Not enough memory for visualization.") end
        T_v    = zeros(nx_v, ny_v, nz_v) # global array for visu
        Vm_v   = zeros(nx_v, ny_v, nz_v)
        Pr_v   = zeros(nx_v, ny_v, nz_v)
        T_inn  = zeros(nx-2, ny-2, nz-2) # no halo local array for visu
        Vm_inn = zeros(nx-2, ny-2, nz-2)
        Pr_inn = zeros(nx-2, ny-2, nz-2)
        xi_g,yi_g,zi_g = LinRange(-lx/2+dx+dx/2,lx/2-dx-dx/2,nx_v),LinRange(-ly/2+dy+dy/2,ly/2-dy-dy/2,ny_v),LinRange(dz+dz/2,lz-dz-dz/2,nz_v) # inner points only
    end
    # initialisation
    for ik=1:2
        AMDGPU.HSA.signal_store_screlease(sig[ik].signal[],1)
    end
    @roc wait=false mark=false signal=sig[1] queue=rocqueues[1] groupsize=threads gridsize=grid compute_phase!(ρgz_c,ph_ice,ph_bed,xc,yc,zc,r_box,w_box,r_rnd,z_bed,δ_sd,ρg0.air,ρg0.ice,ρg0.bed)
    @roc wait=false mark=false signal=sig[2] queue=rocqueues[1] groupsize=threads gridsize=grid init_U!(U,T,ρCp.ice,T0.ice)
    wait(sig[2])
    ρgz .= ameanz(ρgz_c[2:end-1,2:end-1,:])
    iter_evo=Float64[]; errs_evo=ElasticMatrix{Float64}(undef,length(ϵtol),0)
    # time loop
    for it = 1:nt
        me==0 && @printf("it = %d \n",it)
        errs = 2.0.*ϵtol; iter = 1
        resize!(iter_evo,0); resize!(errs_evo,length(ϵtol),0)
        # iteration loop
        while any(errs .>= ϵtol) && iter <= maxiter
            for ik=3:19
                AMDGPU.HSA.signal_store_screlease(sig[ik].signal[],1)
            end
            for ik=1:2, is=1:nstep
                AMDGPU.HSA.signal_store_screlease(sig2[ik,is].signal[],1)
            end
            # mechanics
            @roc wait=false mark=false signal=sig[3] queue=rocqueues[1] groupsize=threads gridsize=grid compute_η!(η,T,Q_R,η0.air,η0.ice,η0.bed,ph_ice,ph_bed)
            ###### hide_comm
            for is=1:2
                @roc wait=false mark=false signal=sig2[1,is] queue=rocqueues[is] groupsize=threads gridsize=grid update_iter_params!(ητ,η,b_width,is)
                if is==1
                    @roc wait=false mark=false signal=sig[4] queue=rocqueues[1] groupsize=threads gridsize=grid bc_x!(ητ)
                    @roc wait=false mark=false signal=sig[5] queue=rocqueues[1] groupsize=threads gridsize=grid bc_y!(ητ)
                    @roc wait=false mark=false signal=sig[6] queue=rocqueues[1] groupsize=threads gridsize=grid bc_z!(ητ)
                end
            end
            wait(sig[6])
            # print("update_halo!(ητ) ... ")
            update_halo!(ητ)
            wait(sig2[1,2])
            # println("done")
            # ###### hide_comm
            @roc wait=false mark=false signal=sig[7] queue=rocqueues[1] groupsize=threads gridsize=grid update_P_ε!(Pr,dPr,εxx,εyy,εzz,εxy,εxz,εyz,Vx,Vy,Vz,∇V,η,r,θ_dτ,dx,dy,dz)
            @roc wait=false mark=false signal=sig[8] queue=rocqueues[1] groupsize=threads gridsize=grid update_τ!(τxx,τyy,τzz,τxy,τxz,τyz,εxx,εyy,εzz,εxy,εxz,εyz,εII,η,dτ_r)
            wait(sig[8])
            # ###### hide_comm
            for is=1:nstep
                @roc wait=false mark=false signal=sig2[2,is] queue=rocqueues[is] groupsize=threads gridsize=grid update_velocities!(Vx,Vy,Vz,Pr,τxx,τyy,τzz,τxy,τxz,τyz,ητ,ρgx,ρgy,ρgz,nudτ,dx,dy,dz,b_width,is)
                if is==1
                    # free slip x
                    @roc wait=false mark=false signal=sig[9]  queue=rocqueues[1] groupsize=threads gridsize=grid bc_y!(Vx)
                    @roc wait=false mark=false signal=sig[10] queue=rocqueues[1] groupsize=threads gridsize=grid bc_z!(Vx)
                    # free slip y
                    @roc wait=false mark=false signal=sig[11] queue=rocqueues[1] groupsize=threads gridsize=grid bc_x!(Vy)
                    @roc wait=false mark=false signal=sig[12] queue=rocqueues[1] groupsize=threads gridsize=grid bc_z!(Vy)
                    # free slip z
                    @roc wait=false mark=false signal=sig[13] queue=rocqueues[1] groupsize=threads gridsize=grid bc_x!(Vz)
                    @roc wait=false mark=false signal=sig[14] queue=rocqueues[1] groupsize=threads gridsize=grid bc_y!(Vz)
                    # free surface top
                    @roc wait=false mark=false signal=sig[15] queue=rocqueues[1] groupsize=threads gridsize=grid bc_Vx!(Vx,η,Pr,ph_ice,ph_bed,dx)
                    @roc wait=false mark=false signal=sig[16] queue=rocqueues[1] groupsize=threads gridsize=grid bc_Vy!(Vy,η,Pr,ph_ice,ph_bed,dy)
                    @roc wait=false mark=false signal=sig[17] queue=rocqueues[1] groupsize=threads gridsize=grid bc_Vz!(Vz,η,Pr,dz)
                    # no slip bottom
                    @roc wait=false mark=false signal=sig[18] queue=rocqueues[1] groupsize=threads gridsize=grid bc_Vxy!(Vx)
                    @roc wait=false mark=false signal=sig[19] queue=rocqueues[1] groupsize=threads gridsize=grid bc_Vxy!(Vy)
                end
            end
            wait(sig[19])
            # print("update_halo!(Vx,Vy,Vz) ... ")
            update_halo!(Vx,Vy,Vz)
            wait(sig2[2,2])
            # println("done")
            ###### hide_comm
            if iter % ncheck == 0
                AMDGPU.HSA.signal_store_screlease(sig[20].signal[],1)
                @roc wait=false mark=false signal=sig[20] queue=rocqueues[1] groupsize=threads gridsize=grid compute_residuals!(r_Vx,r_Vy,r_Vz,Pr,τxx,τyy,τzz,τxy,τxz,τyz,ρgx,ρgy,ρgz,dx,dy,dz)
                wait(sig[20])
                errVx = maximum_g(abs.(r_Vx))
                errVy = maximum_g(abs.(r_Vy))
                errVz = maximum_g(abs.(r_Vz))
                errPr = maximum_g(abs.(dPr[2:end-1,2:end-1,2:end-1]))
                errs  = (errVx,errVy,errVz,errPr)
                push!(iter_evo,iter/min(nx_g(),ny_g(),nz_g())); append!(errs_evo,errs)
                me==0 && @printf("  iter/nz=%.1f,errs=[ %1.3e, %1.3e, %1.3e, %1.3e ] \n",iter/min(nx_g(),ny_g(),nz_g()),errs...)
                # push!(iter_evo,iter/min(nx,ny,nz)); append!(errs_evo,errs)
                # @printf("  iter/nz=%.1f,errs=[ %1.3e, %1.3e, %1.3e, %1.3e ] \n",iter/min(nx,ny,nz),errs...)
            end
            iter += 1
        end
        for ik = 21:25
            AMDGPU.HSA.signal_store_screlease(sig[ik].signal[],1)
        end
        for ik=3, is=1:nstep
            AMDGPU.HSA.signal_store_screlease(sig2[ik,is].signal[],1)
        end
        # thermal
        @roc wait=false mark=false signal=sig[21] queue=rocqueues[1] groupsize=threads gridsize=grid compute_T!(T,U,ρCp.ice)
        @roc wait=false mark=false signal=sig[22] queue=rocqueues[1] groupsize=threads gridsize=grid update_qU!(qUx,qUy,qUz,T,λ.ice,dx,dy,dz)
        wait(sig[22])
        # ###### hide_comm
        for is=1:nstep
            @roc wait=false mark=false signal=sig2[3,is] queue=rocqueues[is] groupsize=threads gridsize=grid update_U!(U,qUx,qUy,qUz,η,εII,dt,dx,dy,dz,b_width,is)
            if is==1
                @roc wait=false mark=false signal=sig[23] queue=rocqueues[1] groupsize=threads gridsize=grid bc_x!(U)
                @roc wait=false mark=false signal=sig[24] queue=rocqueues[1] groupsize=threads gridsize=grid bc_y!(U)
                @roc wait=false mark=false signal=sig[25] queue=rocqueues[1] groupsize=threads gridsize=grid bc_z!(U)
            end
        end
        wait(sig[25])
        # print("update_halo!(U) ... ")
        update_halo!(U)
        wait(sig2[3,2])
        # println("done")
        # ###### hide_comm
    end
    if do_save
        Vmag .= sqrt.(ameanx(Vx).^2 .+ ameany(Vy).^2 .+ ameanz(Vz).^2)
        T_inn  .= Array(T)[2:end-1,2:end-1,2:end-1];    gather!(T_inn , T_v )
        Vm_inn .= Array(Vmag)[2:end-1,2:end-1,2:end-1]; gather!(Vm_inn, Vm_v)
        Pr_inn .= Array(Pr)[2:end-1,2:end-1,2:end-1];   gather!(Pr_inn, Pr_v)
        # T_inn  .= Array(ρgz_c)[2:end-1,2:end-1,2:end-1];    gather!(T_inn , T_v )
        # Vm_inn .= Array(ph_ice)[2:end-1,2:end-1,2:end-1]; gather!(Vm_inn, Vm_v)
        # Pr_inn .= Array(ph_bed)[2:end-1,2:end-1,2:end-1];   gather!(Pr_inn, Pr_v)
        if me==0
            if isdir(outdir)==false mkdir(outdir) end
            save_array(joinpath(outdir,"out_Pr"  ),convert.(Float32,Array(Pr_v)))
            save_array(joinpath(outdir,"out_Vmag"),convert.(Float32,Array(Vm_v)))
            save_array(joinpath(outdir,"out_T"   ),convert.(Float32,Array(T_v) ))
            save_array(joinpath(outdir,"out_xc"  ),convert.(Float32,Array(xi_g)))
            save_array(joinpath(outdir,"out_yc"  ),convert.(Float32,Array(yi_g)))
            save_array(joinpath(outdir,"out_zc"  ),convert.(Float32,Array(zi_g)))
            open(joinpath(outdir,"nxyz.txt"),"w") do io
                println(io,"$(size(Pr_v,1)) $(size(Pr_v,2)) $(size(Pr_v,3))")
            end
        end
    end
    finalize_global_grid()
    return
end

main(;do_save=true)
