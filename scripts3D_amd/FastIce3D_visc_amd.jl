using AMDGPU
using GeometryBasics,LinearAlgebra,ElasticArrays,Printf
# using GLMakie

include("FD_helpers.jl")
const DAT = Float64

@views amean1(A) = 0.5.*(A[1:end-1] .+ A[2:end])
@views ameanx(A) = 0.5.*(A[1:end-1,:,:] .+ A[2:end,:,:])
@views ameany(A) = 0.5.*(A[:,1:end-1,:] .+ A[:,2:end,:])
@views ameanz(A) = 0.5.*(A[:,:,1:end-1] .+ A[:,:,2:end])

function save_array(Aname,A)
    fname = string(Aname,".bin")
    out = open(fname,"w"); write(out,A); close(out)
end

function update_iter_params!(ητ,η)
    @get_thread_idx()
    if (ix<=size(ητ,1)-2 && iy<=size(ητ,2)-2 && iz<=size(ητ,3)-2) @inn(ητ) = @maxloc(η) end
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

function update_U!(U,qUx,qUy,qUz,η,εII,dt,dx,dy,dz)
    @get_thread_idx()
    if (ix<=size(U,1)-2 && iy<=size(U,2)-2 && iz<=size(U,3)-2)
        @inn(U) = @inn(U) + dt*(-(@d_xa(qUx)/dx + @d_ya(qUy)/dy + @d_za(qUz)/dz) + 2.0*@inn(η)*@all(εII)^2)
    end
    return
end

function compute_T!(T,U,ρCp)
    @get_thread_idx()
    if (ix<=size(T,1) && iy<=size(T,2) && iz<=size(T,3)) @all(T) = @all(U)/ρCp end
    return
end

function update_velocities!(Vx,Vy,Vz,Pr,τxx,τyy,τzz,τxy,τxz,τyz,ητ,ρgx,ρgy,ρgz,nudτ,dx,dy,dz)
    @get_thread_idx()
    if (ix<=size(Vx,1)-2 && iy<=size(Vx,2)-2 && iz<=size(Vx,3)-2) @inn(Vx) = @inn(Vx) + (-@d_xi(Pr)/dx + @d_xi(τxx)/dx + @d_ya(τxy)/dy + @d_za(τxz)/dz - @all(ρgx))*nudτ/@av_xi(ητ) end
    if (ix<=size(Vy,1)-2 && iy<=size(Vy,2)-2 && iz<=size(Vy,3)-2) @inn(Vy) = @inn(Vy) + (-@d_yi(Pr)/dy + @d_yi(τyy)/dy + @d_xa(τxy)/dx + @d_za(τyz)/dz - @all(ρgy))*nudτ/@av_yi(ητ) end
    if (ix<=size(Vz,1)-2 && iy<=size(Vz,2)-2 && iz<=size(Vz,3)-2) @inn(Vz) = @inn(Vz) + (-@d_zi(Pr)/dz + @d_zi(τzz)/dz + @d_xa(τxz)/dx + @d_ya(τyz)/dy - @all(ρgz))*nudτ/@av_zi(ητ) end
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

@views function main(;do_visu=false,do_save=false)
    println("Process selecting device $(AMDGPU.default_device_id())")
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
    threads    = (32,2,2)
    nz         = 2*32-1
    nx         = ceil(Int,(nz+1)*lx/lz)-1
    ny         = ceil(Int,(nz+1)*ly/lz)-1
    println("Grid nx=$nx, ny=$ny, nz=$nz")
    grid       = (nx+1,ny+1,nz+1)
    ϵtol       = (1e-6,1e-6,1e-6,1e-6)
    maxiter    = 100min(nx,ny,nz)
    ncheck     = ceil(Int,5min(nx,ny,nz))
    r          = 0.6
    re_mech    = 5.2π
    nt         = 10
    # preprocessing
    dx,dy,dz   = lx/nx,ly/ny,lz/nz
    xv,yv,zv   = LinRange(-lx/2,lx/2,nx+1),LinRange(-ly/2,ly/2,ny+1),LinRange(0,lz,nz+1)
    xc,yc,zc   = amean1(xv),amean1(yv),amean1(zv)
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
    # 
    nstep = 1 # 1 normal or 2 if comm overlap
    # nkern = 1
    rocqueues = Vector{AMDGPU.ROCQueue}(undef,nstep)
    for istep = 1:nstep
        rocqueues[istep] = istep == 1 ? ROCQueue(AMDGPU.default_device(); priority=:high) : ROCQueue(AMDGPU.default_device())
    end
    # signals = Array{AMDGPU.ROCKernelSignal}(undef,nkern,nstep)
    # sig_real = Array{AMDGPU.ROCSignal}(undef,nkern,nstep)
    # for istep = 1:nstep, for ikern = 1:nkern
    #     sig_real[ikern,istep] = ROCSignal()
    # end
    # initialisation
    wait( @roc queue=rocqueues[1] groupsize=threads gridsize=grid compute_phase!(ρgz_c,ph_ice,ph_bed,xc,yc,zc,r_box,w_box,r_rnd,z_bed,δ_sd,ρg0.air,ρg0.ice,ρg0.bed) )
    wait( @roc queue=rocqueues[1] groupsize=threads gridsize=grid init_U!(U,T,ρCp.ice,T0.ice) )
    ρgz .= ameanz(ρgz_c[2:end-1,2:end-1,:])
    iter_evo=Float64[]; errs_evo=ElasticMatrix{Float64}(undef,length(ϵtol),0)
    # time loop
    for it = 1:nt
        @printf("it = %d \n",it)
        errs = 2.0.*ϵtol; iter = 1
        resize!(iter_evo,0); resize!(errs_evo,length(ϵtol),0)
        # iteration loop
        while any(errs .>= ϵtol) && iter <= maxiter
            # for istep = 1:nstep, for ikern = 1:nkern
            #     AMDGPU.HSA.signal_store_screlease(sig_real[ikern,istep].signal[], 1)
            # end
            # mechanics
            wait( @roc queue=rocqueues[1] groupsize=threads gridsize=grid compute_η!(η,T,Q_R,η0.air,η0.ice,η0.bed,ph_ice,ph_bed) )
            wait( @roc queue=rocqueues[1] groupsize=threads gridsize=grid update_iter_params!(ητ,η) )
            wait( @roc queue=rocqueues[1] groupsize=threads gridsize=grid bc_x!(ητ) )
            wait( @roc queue=rocqueues[1] groupsize=threads gridsize=grid bc_y!(ητ) )
            wait( @roc queue=rocqueues[1] groupsize=threads gridsize=grid bc_z!(ητ) )

            # for istep=1:2
            #     signals[1,istep] = @roc wait=false mark=false signal=sig_real[1,istep] queue=rocqueues[istep] groupsize=threads gridsize=grid update_iter_params!(ητ,η)
            # end
            # signals[2,1] = @roc wait=false mark=false signal=sig_real[2,1] queue=rocqueues[1] groupsize=threads gridsize=grid bc_x!(ητ)
            # signals[3,1] = @roc wait=false mark=false signal=sig_real[1,1] queue=rocqueues[1] groupsize=threads gridsize=grid bc_y!(ητ)
            # signals[4,1] = @roc wait=false mark=false signal=sig_real[2,1] queue=rocqueues[1] groupsize=threads gridsize=grid bc_z!(ητ)
            # for ikern=1:4
            #     wait(signals[ikern,1])
            # end
            # # update_halo!(ητ)
            # wait(signals[1,2])

            wait( @roc queue=rocqueues[1] groupsize=threads gridsize=grid update_P_ε!(Pr,dPr,εxx,εyy,εzz,εxy,εxz,εyz,Vx,Vy,Vz,∇V,η,r,θ_dτ,dx,dy,dz) )
            wait( @roc queue=rocqueues[1] groupsize=threads gridsize=grid update_τ!(τxx,τyy,τzz,τxy,τxz,τyz,εxx,εyy,εzz,εxy,εxz,εyz,εII,η,dτ_r) )

            wait( @roc queue=rocqueues[1] groupsize=threads gridsize=grid update_velocities!(Vx,Vy,Vz,Pr,τxx,τyy,τzz,τxy,τxz,τyz,ητ,ρgx,ρgy,ρgz,nudτ,dx,dy,dz) )
            wait( @roc queue=rocqueues[1] groupsize=threads gridsize=grid bc_y!(Vx) ) # free slip x
            wait( @roc queue=rocqueues[1] groupsize=threads gridsize=grid bc_z!(Vx) )
            wait( @roc queue=rocqueues[1] groupsize=threads gridsize=grid bc_x!(Vy) ) # free slip y
            wait( @roc queue=rocqueues[1] groupsize=threads gridsize=grid bc_z!(Vy) )
            wait( @roc queue=rocqueues[1] groupsize=threads gridsize=grid bc_x!(Vz) ) # free slip z
            wait( @roc queue=rocqueues[1] groupsize=threads gridsize=grid bc_y!(Vz) )
            wait( @roc queue=rocqueues[1] groupsize=threads gridsize=grid bc_Vx!(Vx,η,Pr,ph_ice,ph_bed,dx) ) # free surface top
            wait( @roc queue=rocqueues[1] groupsize=threads gridsize=grid bc_Vy!(Vy,η,Pr,ph_ice,ph_bed,dy) )
            wait( @roc queue=rocqueues[1] groupsize=threads gridsize=grid bc_Vz!(Vz,η,Pr,dz) )
            wait( @roc queue=rocqueues[1] groupsize=threads gridsize=grid bc_Vxy!(Vx) ) # no slip bottom
            wait( @roc queue=rocqueues[1] groupsize=threads gridsize=grid bc_Vxy!(Vy) )
            if iter % ncheck == 0
                wait( @roc queue=rocqueues[1] groupsize=threads gridsize=grid compute_residuals!(r_Vx,r_Vy,r_Vz,Pr,τxx,τyy,τzz,τxy,τxz,τyz,ρgx,ρgy,ρgz,dx,dy,dz) )
                errs = maximum.((abs.(r_Vx),abs.(r_Vy),abs.(r_Vz),abs.(dPr[2:end-1,2:end-1,2:end-1])))
                push!(iter_evo,iter/min(nx,ny,nz)); append!(errs_evo,errs)
                @printf("  iter/nz=%.3f,errs=[ %1.3e, %1.3e, %1.3e, %1.3e ] \n",iter/min(nx,ny,nz),errs...)
            end
            iter += 1
        end
        # thermal
        wait( @roc queue=rocqueues[1] groupsize=threads gridsize=grid compute_T!(T,U,ρCp.ice) )
        wait( @roc queue=rocqueues[1] groupsize=threads gridsize=grid update_qU!(qUx,qUy,qUz,T,λ.ice,dx,dy,dz) )
        wait( @roc queue=rocqueues[1] groupsize=threads gridsize=grid update_U!(U,qUx,qUy,qUz,η,εII,dt,dx,dy,dz) )
        wait( @roc queue=rocqueues[1] groupsize=threads gridsize=grid bc_x!(U) )
        wait( @roc queue=rocqueues[1] groupsize=threads gridsize=grid bc_y!(U) )
        wait( @roc queue=rocqueues[1] groupsize=threads gridsize=grid bc_z!(U) )
    end
    if do_visu
        # visualisation
        Vmag .= sqrt.(ameanx(Vx).^2 .+ ameany(Vy).^2 .+ ameanz(Vz).^2)
        fig = Figure(resolution=(3000,800),fontsize=32)
        axs = (
            Pr   = Axis3(fig[1,1][1,1][1,1];aspect=:data,xlabel="x",ylabel="y",zlabel="z",title="Pr"),
            Vmag = Axis3(fig[1,1][1,2][1,1];aspect=:data,xlabel="x",ylabel="y",zlabel="z",title="|V|"),
            T    = Axis3(fig[1,1][1,3][1,1];aspect=:data,xlabel="x",ylabel="y",zlabel="z",title="T"),
        )
        plts = (
            Pr   = volumeslices!(axs.Pr  ,xc,yc,zc,Array(Pr  );colormap=:turbo),
            Vmag = volumeslices!(axs.Vmag,xc,yc,zc,Array(Vmag);colormap=:turbo),
            T    = volumeslices!(axs.T   ,xc,yc,zc,Array(T   );colormap=:turbo),
        )
        sgrid = SliderGrid(
            fig[2,1],
            (label = "yz plane - x axis", range = 1:length(xc)),
            (label = "xz plane - y axis", range = 1:length(yc)),
            (label = "xy plane - z axis", range = 1:length(zc)),
        )
        # connect sliders to `volumeslices` update methods
        sl_yz, sl_xz, sl_xy = sgrid.sliders
        on(sl_yz.value) do v; for prop in eachindex(plts) plts[prop][:update_yz][](v) end; end
        on(sl_xz.value) do v; for prop in eachindex(plts) plts[prop][:update_xz][](v) end; end
        on(sl_xy.value) do v; for prop in eachindex(plts) plts[prop][:update_xy][](v) end; end
        set_close_to!(sl_yz, .5length(xc))
        set_close_to!(sl_xz, .5length(yc))
        set_close_to!(sl_xy, .5length(zc))
        [Colorbar(fig[1,1][irow,icol][1,2],plts[(irow-1)*2+icol]) for irow in 1:1,icol in 1:3]
        display(fig)
    end
    if do_save
        if isdir("out")==false mkdir("out") end
        Vmag .= sqrt.(ameanx(Vx).^2 .+ ameany(Vy).^2 .+ ameanz(Vz).^2)
        save_array("out/out_Pr"  ,convert.(Float32,Array(Pr)  ))
        save_array("out/out_Vmag",convert.(Float32,Array(Vmag)))
        save_array("out/out_T"   ,convert.(Float32,Array(T)   ))
        save_array("out/out_xc"  ,convert.(Float32,Array(xc)  ))
        save_array("out/out_yc"  ,convert.(Float32,Array(yc)  ))
        save_array("out/out_zc"  ,convert.(Float32,Array(zc)  ))
    end
    return
end

main(;do_visu=false,do_save=true)
