using AMDGPU
using GeometryBasics,LinearAlgebra,ElasticArrays,Printf
using GLMakie

include("FD_helpers.jl")
const DAT = Float64

@views amean1(A) = 0.5.*(A[1:end-1] .+ A[2:end])
@views ameanx(A) = 0.5.*(A[1:end-1,:,:] .+ A[2:end,:,:])
@views ameany(A) = 0.5.*(A[:,1:end-1,:] .+ A[:,2:end,:])
@views ameanz(A) = 0.5.*(A[:,:,1:end-1] .+ A[:,:,2:end])

function update_iter_params!(ητ,η,nx,ny,nz)
    @get_thread_idx()
    (ix<=nx-2 && iy<=ny-2 && iz<=nz-2) || return
    @inn(ητ) = @maxloc(η)
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

function update_normal_τ!(Pr,dPr,εxx,εyy,εzz,εxy,εxz,εyz,Vx,Vy,Vz,∇V,η,r,θ_dτ,dx,dy,dz,nx,ny,nz)
    @get_thread_idx()
    (ix<=nx && iy<=ny && iz<=nz) || return
    @all(∇V)  = @d_xa(Vx)/dx + @d_ya(Vy)/dy + @d_za(Vz)/dz
    @all(dPr) = -@all(∇V)
    @all(Pr)  = @all(Pr) + @all(dPr)*@all(η)*r/θ_dτ
    @all(εxx) = @d_xa(Vx)/dx - @all(∇V)/3.0
    @all(εyy) = @d_ya(Vy)/dy - @all(∇V)/3.0
    @all(εzz) = @d_za(Vz)/dz - @all(∇V)/3.0
    @all(εxy) = 0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx)
    @all(εxz) = 0.5*(@d_zi(Vx)/dz + @d_xi(Vz)/dx)
    @all(εyz) = 0.5*(@d_zi(Vy)/dz + @d_yi(Vz)/dy)
    return
end

function update_shear_τ!(τxx,τyy,τzz,τxy,τxz,τyz,εxx,εyy,εzz,εxy,εxz,εyz,εII,η,dτ_r,nx,ny,nz)
    @get_thread_idx()
    (ix<=nx && iy<=ny && iz<=nz) || return
    @all(τxx) = @all(τxx) + (-@all(τxx) + 2.0*@all(η)*@all(εxx))*dτ_r
    @all(τyy) = @all(τyy) + (-@all(τyy) + 2.0*@all(η)*@all(εyy))*dτ_r
    @all(τzz) = @all(τzz) + (-@all(τzz) + 2.0*@all(η)*@all(εzz))*dτ_r
    @all(τxy) = @all(τxy) + (-@all(τxy) + 2.0*@av_xyi(η)*@all(εxy))*dτ_r
    @all(τxz) = @all(τxz) + (-@all(τxz) + 2.0*@av_xzi(η)*@all(εxz))*dτ_r
    @all(τyz) = @all(τyz) + (-@all(τyz) + 2.0*@av_yzi(η)*@all(εyz))*dτ_r
    @all(εII) = sqrt(0.5*(@inn(εxx)^2 + @inn(εyy)^2 + @inn(εzz)^2) + @av_xya(εxy)^2 + @av_xza(εxz)^2 + @av_yza(εyz)^2)
    return
end

function update_qU!(qUx,qUy,qUz,T,λ,dx,dy,dz,nx,ny,nz)
    @get_thread_idx()
    (ix<=nx && iy<=ny && iz<=nz) || return
    @all(qUx) = -λ*@d_xi(T)/dx
    @all(qUy) = -λ*@d_yi(T)/dy
    @all(qUz) = -λ*@d_zi(T)/dz
    return
end

function update_U!(U,qUx,qUy,qUz,η,εII,dt,dx,dy,dz,nx,ny,nz)
    @get_thread_idx()
    (ix<=nx && iy<=ny && iz<=nz) || return
    @inn(U) = @inn(U) + dt*(-(@d_xa(qUx)/dx + @d_ya(qUy)/dy + @d_za(qUz)/dz) + 2.0*@inn(η)*@all(εII)^2)
    return
end

function compute_T!(T,U,ρCp,nx,ny,nz)
    @get_thread_idx()
    (ix<=nx && iy<=ny && iz<=nz) || return
    @all(T) = @all(U)/ρCp
    return
end

function update_velocities!(Vx,Vy,Vz,Pr,τxx,τyy,τzz,τxy,τxz,τyz,ητ,ρgx,ρgy,ρgz,nudτ,dx,dy,dz,nx,ny,nz)
    @get_thread_idx()
    (ix<=nx && iy<=ny && iz<=nz) || return
    @inn(Vx) = @inn(Vx) + (-@d_xi(Pr)/dx + @d_xi(τxx)/dx + @d_ya(τxy)/dy + @d_za(τxz)/dz - @all(ρgx))*nudτ/@av_xi(ητ)
    @inn(Vy) = @inn(Vy) + (-@d_yi(Pr)/dy + @d_yi(τyy)/dy + @d_xa(τxy)/dx + @d_za(τyz)/dz - @all(ρgy))*nudτ/@av_yi(ητ)
    @inn(Vz) = @inn(Vz) + (-@d_zi(Pr)/dz + @d_zi(τzz)/dz + @d_xa(τxz)/dx + @d_ya(τyz)/dy - @all(ρgz))*nudτ/@av_zi(ητ)
    return
end

function bc_Vxy!(Vxy)
    @get_thread_idx()
    nx,ny,nz=size(Vxy)
    if (ix==1 && iy<=ny && iz<=nz) Vxy[ix,iy,ix] = -Vxy[ix,iy,ix+1] end
    return
end

function bc_Vz!(Vz,η,Pr,dz)
    @get_thread_idx()
    nx,ny,nz=size(Vz)
    if (ix<=nz && iy<=ny && iz==nz) Vz[ix,iy,iz] = Vz[ix,iy,iz-1] + 0.5*dz/η[ix,iy,iz]*(Pr[ix,iy,iz] + 1.0/3.0*(-Pr[ix,iy,iz-1] + 2.0*η[ix,iy,iz-1]*(Vz[ix,iy,iz-1] - Vz[ix,iy,iz-2])/dz)) end
    return
end

@parallel function compute_residuals!(r_Vx,r_Vy,r_Vz,Pr,τxx,τyy,τzz,τxy,τxz,τyz,ρgx,ρgy,ρgz,dx,dy,dz,nx,ny,nz)
    @get_thread_idx()
    (ix<=nx && iy<=ny && iz<=nz) || return
    @all(r_Vx) = -@d_xi(Pr)/dx + @d_xi(τxx)/dx + @d_ya(τxy)/dy + @d_za(τxz)/dz - @all(ρgx)
    @all(r_Vy) = -@d_yi(Pr)/dy + @d_yi(τyy)/dy + @d_xa(τxy)/dx + @d_za(τyz)/dz - @all(ρgy)
    @all(r_Vz) = -@d_zi(Pr)/dz + @d_zi(τzz)/dz + @d_xa(τxz)/dx + @d_ya(τyz)/dy - @all(ρgz)
    return
end

function compute_η!(η,T,Q_R,η0_air,η0_ice,η0_bed,ph_ice,ph_bed,nx,ny,nz)
    @get_thread_idx()
    (ix<=nx && iy<=ny && iz<=nz) || return
    t_ice   = @all(ph_ice)
    t_bed   = @all(ph_bed)
    t_air   = 1.0 - t_ice - t_bed
    η_ice   = η0_ice*exp(Q_R/@all(T))
    η_bed   = η0_bed
    η_air   = η0_air
    @all(η) = t_ice*η_ice + t_bed*η_bed + t_air*η_air
    return
end

@inline function sd_round_box(p,b,r)
  q = abs.(p) .- b
  return norm(max.(q,0.0)) + min(max(q[1],max(q[2],q[3])),0.0) - r
end

function compute_phase!(ρgz_c,ph_ice,ph_bed,xc,yc,zc,r_box,w_box,r_rnd,z_bed,δ_sd,ρg0_air,ρg0_ice,ρg0_bed,nx,ny,nz)
    @get_thread_idx()
    (ix<=nx && iy<=ny && iz<=nz) || return
    sd_bed = zc[iz]-z_bed
    sd_ice = sd_round_box(Point(xc[ix],yc[iy],zc[iz])-r_box,w_box,r_rnd)
    sd_ice = max(sd_ice,-sd_bed)
    t_ice  = 0.5*(tanh(-sd_ice/δ_sd) + 1.0)
    t_bed  = 0.5*(tanh(-sd_bed/δ_sd) + 1.0)
    t_air  = 1.0 - t_ice - t_bed
    @all(ρgz_c)  = t_ice*ρg0_ice + t_bed*ρg0_bed + t_air*ρg0_air
    @all(ph_ice) = t_ice
    @all(ph_bed) = t_bed
    return
end

function init_U!(U,T,ρCp,T0,nx,ny,nz)
    @get_thread_idx()
    (ix<=nx && iy<=ny && iz<=nz) || return
    @all(T) = T0
    @all(U) = ρCp*@all(T)
    return
end

@views function main(;do_visu=false,do_save=false)
    println("Process selecting device $(AMDGPU.default_device_id())")
    # physics
    lx,ly,lz   = 40.0,40.0,10.0
    η0         = (ice = 1.0  ,bed = 1e2  ,air = 1e-8 )
    ρg0        = (ice = 1.0  ,bed = 1.0  ,air = 0.0  )
    λ          = (ice = 1.0  ,bed = 1.0  ,air = 1.0  )
    ρCp        = (ice = 1.0  ,bed = 1.0  ,air = 1.0  )
    T0         = (ice = 253.0,bed = 253.0,air = 253.0)
    Q_R        = 10.0
    r_box      = Vec(0.0lx,0.0ly,0.3lz)
    w_box      = Vec(0.5lx-0.6lz,0.5ly-0.6lz,0.3lz)
    r_rnd      = 0.25lz
    z_bed      = 0.1lz
    # numerics
    threads    = (32,2,2)
    nz         = 32
    nx         = ceil(Int,nz*lx/lz)
    ny         = ceil(Int,nz*ly/lz)
    grid       = (nx,ny,nz)
    ϵtol       = (1e-6,1e-6,1e-6,1e-6)
    maxiter    = 100min(nx,ny,nz)
    ncheck     = ceil(Int,5min(nx,ny,nz))
    r          = 0.6
    re_mech    = 3π
    nt         = 2
    # preprocessing
    dx,dy,dz   = lx/nx,ly/ny,lz/nz
    xv,yv,zv   = LinRange(-lx/2,lx/2,nx+1),LinRange(-ly/2,ly/2,ny+1),LinRange(0,lz,nz+1)
    xc,yc,zc   = amean1(xv),amean1(yv),amean1(zv)
    lτ         = min(lx,ly,lz)
    vdτ        = min(dx,dy,dz)/sqrt(3.1)
    θ_dτ       = lτ*(r+2.0)/(re_mech*vdτ)
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
    # initialisation
    compute_phase!(ρgz_c,ph_ice,ph_bed,xc,yc,zc,r_box,w_box,r_rnd,z_bed,δ_sd,ρg0.air,ρg0.ice,ρg0.bed)
    init_U!(U,T,ρCp.ice,T0.ice)
    ρgz .= ameanz(ρgz_c[2:end-1,2:end-1,:])
    iter_evo=Float64[]; errs_evo=ElasticMatrix{Float64}(undef,length(ϵtol),0)
    # time loop
    for it = 1:nt
        @printf("it = %d \n",it)
        errs = 2.0.*ϵtol; iter = 1
        resize!(iter_evo,0); resize!(errs_evo,length(ϵtol),0)
        # iteration loop
        while any(errs .>= ϵtol) && iter <= maxiter
            # mechanics
            compute_η!(η,T,Q_R,η0.air,η0.ice,η0.bed,ph_ice,ph_bed)
            update_iter_params!(ητ,η)
            bc_x!(ητ)
            bc_y!(ητ)
            bc_z!(ητ)
            update_normal_τ!(Pr,dPr,εxx,εyy,εzz,εxy,εxz,εyz,Vx,Vy,Vz,∇V,η,r,θ_dτ,dx,dy,dz)
            update_shear_τ!(τxx,τyy,τzz,τxy,τxz,τyz,εxx,εyy,εzz,εxy,εxz,εyz,εII,η,dτ_r)
            update_velocities!(Vx,Vy,Vz,Pr,τxx,τyy,τzz,τxy,τxz,τyz,ητ,ρgx,ρgy,ρgz,nudτ,dx,dy,dz)
            # free slip x
            bc_y!(Vx)
            bc_z!(Vx)
            # free slip y
            bc_x!(Vy)
            bc_z!(Vy)
            # free slip z
            bc_x!(Vz)
            bc_y!(Vz)
            # no slip bottom
            bc_Vxy!(Vx)
            bc_Vxy!(Vy)
            # free surface top
            bc_Vz!(Vz,η,Pr,dz)
            if iter % ncheck == 0
                compute_residuals!(r_Vx,r_Vy,r_Vz,Pr,τxx,τyy,τzz,τxy,τxz,τyz,ρgx,ρgy,ρgz,dx,dy,dz)
                errs = maximum.((abs.(r_Vx),abs.(r_Vy),abs.(r_Vz),abs.(dPr)))
                push!(iter_evo,iter/min(nx,ny,nz));append!(errs_evo,errs)
                @printf("  iter/nz=%.3f,errs=[ %1.3e, %1.3e, %1.3e, %1.3e ] \n",iter/min(nx,ny,nz),errs...)
            end
            iter += 1
        end
        # thermal
        compute_T!(T,U,ρCp.ice)
        update_qU!(qUx,qUy,qUz,T,λ.ice,dx,dy,dz)
        update_U!(U,qUx,qUy,qUz,η,εII,dt,dx,dy,dz)
        bc_x!(U)
        bc_y!(U)
        bc_z!(U)
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
    return
end

main(;do_visu=false,do_save=true)
