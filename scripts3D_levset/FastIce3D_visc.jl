const USE_GPU = haskey(ENV,"USE_GPU") ? parse(Bool,ENV["USE_GPU"]) : true
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA,Float64,3)
else
    @init_parallel_stencil(Threads,Float64,3)
end
using GeometryBasics,LinearAlgebra,ElasticArrays,Printf
using GLMakie

@views amean1(A) = 0.5.*(A[1:end-1] .+ A[2:end])
@views ameanx(A) = 0.5.*(A[1:end-1,:,:] .+ A[2:end,:,:])
@views ameany(A) = 0.5.*(A[:,1:end-1,:] .+ A[:,2:end,:])
@views ameanz(A) = 0.5.*(A[:,:,1:end-1] .+ A[:,:,2:end])

macro my_maxloc(A) esc(:( max.($A[$ixi-1,$iyi-1,$izi-1],$A[$ixi-1,$iyi,$izi-1],$A[$ixi-1,$iyi+1,$izi-1],
                               $A[$ixi  ,$iyi-1,$izi-1],$A[$ixi  ,$iyi,$izi-1],$A[$ixi  ,$iyi+1,$izi-1],
                               $A[$ixi+1,$iyi-1,$izi-1],$A[$ixi+1,$iyi,$izi-1],$A[$ixi+1,$iyi+1,$izi-1],
                               $A[$ixi-1,$iyi-1,$izi-1],$A[$ixi-1,$iyi,$izi-1],$A[$ixi-1,$iyi+1,$izi  ],
                               $A[$ixi  ,$iyi-1,$izi-1],$A[$ixi  ,$iyi,$izi-1],$A[$ixi  ,$iyi+1,$izi  ],
                               $A[$ixi+1,$iyi-1,$izi-1],$A[$ixi+1,$iyi,$izi-1],$A[$ixi+1,$iyi+1,$izi  ],
                               $A[$ixi-1,$iyi-1,$izi-1],$A[$ixi-1,$iyi,$izi-1],$A[$ixi-1,$iyi+1,$izi+1],
                               $A[$ixi  ,$iyi-1,$izi-1],$A[$ixi  ,$iyi,$izi-1],$A[$ixi  ,$iyi+1,$izi+1],
                               $A[$ixi+1,$iyi-1,$izi-1],$A[$ixi+1,$iyi,$izi-1],$A[$ixi+1,$iyi+1,$izi+1]) )) end

import ParallelStencil: INDICES
ix,iy,iz = INDICES
ixi,iyi,izi = :($ix+1), :($iy+1), :($iz+1)

@parallel function update_iter_params!(ητ,η)
    @inn(ητ) = @my_maxloc(η)
    return
end

@parallel_indices (ix,iy) function bc_z!(A)
    A[ix,iy,1  ] = A[ix,iy,2    ]
    A[ix,iy,end] = A[ix,iy,end-1]
    return
end

@parallel_indices (ix,iz) function bc_y!(A)
    A[ix,1  ,iz] = A[ix,2    ,iz]
    A[ix,end,iz] = A[ix,end-1,iz]
    return
end

@parallel_indices (iy,iz) function bc_x!(A)
    A[1  ,iy,iz] = A[2    ,iy,iz]
    A[end,iy,iz] = A[end-1,iy,iz]
    return
end

@parallel function update_normal_τ!(Pr,dPr,εxx,εyy,εzz,εxy,εxz,εyz,Vx,Vy,Vz,∇V,η,r,θ_dτ,dx,dy,dz)
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

@parallel function update_shear_τ!(τxx,τyy,τzz,τxy,τxz,τyz,εxx,εyy,εzz,εxy,εxz,εyz,εII,η,dτ_r)
    @all(τxx) = @all(τxx) + (-@all(τxx) + 2.0*@all(η)*@all(εxx))*dτ_r
    @all(τyy) = @all(τyy) + (-@all(τyy) + 2.0*@all(η)*@all(εyy))*dτ_r
    @all(τzz) = @all(τzz) + (-@all(τzz) + 2.0*@all(η)*@all(εzz))*dτ_r
    @all(τxy) = @all(τxy) + (-@all(τxy) + 2.0*@av_xyi(η)*@all(εxy))*dτ_r
    @all(τxz) = @all(τxz) + (-@all(τxz) + 2.0*@av_xzi(η)*@all(εxz))*dτ_r
    @all(τyz) = @all(τyz) + (-@all(τyz) + 2.0*@av_yzi(η)*@all(εyz))*dτ_r
    @all(εII) = sqrt(0.5*(@inn(εxx)^2 + @inn(εyy)^2 + @inn(εzz)^2) + @av_xya(εxy)^2 + @av_xza(εxz)^2 + @av_yza(εyz)^2)
    return
end

@parallel function update_qU!(qUx,qUy,qUz,T,λ,dx,dy,dz)
    @all(qUx) = -λ*@d_xi(T)/dx
    @all(qUy) = -λ*@d_yi(T)/dy
    @all(qUz) = -λ*@d_zi(T)/dz
    return
end

@parallel function update_U!(U,qUx,qUy,qUz,η,εII,dt,dx,dy,dz)
    @inn(U) = @inn(U) + dt*(-(@d_xa(qUx)/dx + @d_ya(qUy)/dy + @d_za(qUz)/dz) + 2.0*@inn(η)*@all(εII)^2)
    return
end

@parallel function compute_T!(T,U,ρCp)
    @all(T) = @all(U)/ρCp
    return
end

@parallel function update_velocities!(Vx,Vy,Vz,Pr,τxx,τyy,τzz,τxy,τxz,τyz,ητ,ρgx,ρgy,ρgz,nudτ,dx,dy,dz)
    @inn(Vx) = @inn(Vx) + (-@d_xi(Pr)/dx + @d_xi(τxx)/dx + @d_ya(τxy)/dy + @d_za(τxz)/dz - @all(ρgx))*nudτ/@av_xi(ητ)
    @inn(Vy) = @inn(Vy) + (-@d_yi(Pr)/dy + @d_yi(τyy)/dy + @d_xa(τxy)/dx + @d_za(τyz)/dz - @all(ρgy))*nudτ/@av_yi(ητ)
    @inn(Vz) = @inn(Vz) + (-@d_zi(Pr)/dz + @d_zi(τzz)/dz + @d_xa(τxz)/dx + @d_ya(τyz)/dy - @all(ρgz))*nudτ/@av_zi(ητ)
    return
end

@parallel_indices (ix,iy) function bc_Vxy!(Vxy)
    Vxy[ix,iy,1] = -Vxy[ix,iy,2]
    return
end

@parallel_indices (ix,iy) function bc_Vz!(Vz,η,Pr,dz)
    Vz[ix,iy,end] = Vz[ix,iy,end-1] + 0.5*dz/η[ix,iy,end]*(Pr[ix,iy,end-1] + Pr[ix,iy,end] - 2.0*η[ix,iy,end-1]*(Vz[ix,iy,end-1] - Vz[ix,iy,end-2])/dz)
    return
end

@parallel_indices (iy,iz) function bc_Vx!(Vx,η,Pr,ph_ice,ph_bed,dx)
    t_air         = 1.0 - 0.5*(ph_ice[1,iy,iz] + ph_ice[2,iy,iz]) - 0.5*(ph_bed[1,iy,iz] + ph_bed[2,iy,iz])
    Vx[1  ,iy,iz] = t_air*(Vx[2    ,iy,iz] - 0.5*dx/η[1  ,iy,iz]*(Pr[2    ,iy,iz] + Pr[1  ,iy,iz] - 2.0*η[2    ,iy,iz]*(Vx[3    ,iy,iz] - Vx[2    ,iy,iz])/dx)) - (1.0-t_air)*Vx[3    ,iy,iz]
    t_air         = 1.0 - 0.5*(ph_ice[end-1,iy,iz] + ph_ice[end,iy,iz]) - 0.5*(ph_bed[end-1,iy,iz] + ph_bed[end,iy,iz])
    Vx[end,iy,iz] = t_air*(Vx[end-1,iy,iz] + 0.5*dx/η[end,iy,iz]*(Pr[end-1,iy,iz] + Pr[end,iy,iz] - 2.0*η[end-1,iy,iz]*(Vx[end-1,iy,iz] - Vx[end-2,iy,iz])/dx)) - (1.0-t_air)*Vx[end-2,iy,iz]
    return
end

@parallel_indices (ix,iz) function bc_Vy!(Vy,η,Pr,ph_ice,ph_bed,dy)
    t_air         = 1.0 - 0.5*(ph_ice[ix,1,iz] + ph_ice[ix,2,iz]) - 0.5*(ph_bed[ix,1,iz] + ph_bed[ix,2,iz])
    Vy[ix,1  ,iz] = t_air*(Vy[ix,2    ,iz] - 0.5*dy/η[ix,1  ,iz]*(Pr[ix,2    ,iz] + Pr[ix,1  ,iz] - 2.0*η[ix,2    ,iz]*(Vy[ix,3    ,iz] - Vy[ix,2    ,iz])/dy)) - (1.0-t_air)*Vy[ix,3    ,iz]
    t_air         = 1.0 - 0.5*(ph_ice[ix,end-1,iz] + ph_ice[ix,end,iz]) - 0.5*(ph_bed[ix,end-1,iz] + ph_bed[ix,end,iz])
    Vy[ix,end,iz] = t_air*(Vy[ix,end-1,iz] + 0.5*dy/η[ix,end,iz]*(Pr[ix,end-1,iz] + Pr[ix,end,iz] - 2.0*η[ix,end-1,iz]*(Vy[ix,end-1,iz] - Vy[ix,end-2,iz])/dy)) - (1.0-t_air)*Vy[ix,end-2,iz]
    return
end

@parallel function compute_residuals!(r_Vx,r_Vy,r_Vz,Pr,τxx,τyy,τzz,τxy,τxz,τyz,ρgx,ρgy,ρgz,dx,dy,dz)
    @all(r_Vx) = -@d_xi(Pr)/dx + @d_xi(τxx)/dx + @d_ya(τxy)/dy + @d_za(τxz)/dz - @all(ρgx)
    @all(r_Vy) = -@d_yi(Pr)/dy + @d_yi(τyy)/dy + @d_xa(τxy)/dx + @d_za(τyz)/dz - @all(ρgy)
    @all(r_Vz) = -@d_zi(Pr)/dz + @d_zi(τzz)/dz + @d_xa(τxz)/dx + @d_ya(τyz)/dy - @all(ρgz)
    return
end

@parallel_indices (ix,iy,iz) function compute_η!(η,T,Q_R,η0_air,η0_ice,η0_bed,ph_ice,ph_bed)
    t_ice        = ph_ice[ix,iy,iz]
    t_bed        = ph_bed[ix,iy,iz]
    t_air        = 1.0 - t_ice - t_bed
    η_ice        = η0_ice*exp(Q_R/T[ix,iy,iz])
    η_bed        = η0_bed
    η_air        = η0_air
    η[ix,iy,iz]  = t_ice*η_ice + t_bed*η_bed + t_air*η_air
    return
end

@inline function sd_round_box(p,b,r)
  q = abs.(p) .- b
  return norm(max.(q,0.0)) + min(max(q[1],max(q[2],q[3])),0.0) - r
end

@parallel_indices (ix,iy,iz) function compute_phase!(ρgz_c,ph_ice,ph_bed,xc,yc,zc,r_box,w_box,r_rnd,z_bed,δ_sd,ρg0_air,ρg0_ice,ρg0_bed)
    nor    = normalize(Vec3(1.0,1.0,30.0))
    P      = Point(xc[ix],yc[iy],zc[iz])
    sd_bed = dot(nor,P-Point(0,0,z_bed))
    sd_ice = sd_round_box(P-r_box,w_box,r_rnd)
    sd_ice = max(sd_ice,-sd_bed)
    t_ice  = 0.5*(tanh(-sd_ice/δ_sd) + 1.0)
    t_bed  = 0.5*(tanh(-sd_bed/δ_sd) + 1.0)
    t_air  = 1.0 - t_ice - t_bed
    ρgz_c[ix,iy,iz]  = t_ice*ρg0_ice + t_bed*ρg0_bed + t_air*ρg0_air
    ph_ice[ix,iy,iz] = t_ice
    ph_bed[ix,iy,iz] = t_bed
    return
end

@parallel function init_U!(U,T,ρCp,T0)
    @all(T) = T0
    @all(U) = ρCp*@all(T)
    return
end

@views function main()
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
    nz         = 32
    nx         = ceil(Int,nz*lx/lz)
    ny         = ceil(Int,nz*ly/lz)
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
    Vx         = @zeros(nx+1,ny  ,nz  )
    Vy         = @zeros(nx  ,ny+1,nz  )
    Vz         = @zeros(nx  ,ny  ,nz+1)
    Pr         = @zeros(nx  ,ny  ,nz  )
    ∇V         = @zeros(nx  ,ny  ,nz  )
    τxx        = @zeros(nx  ,ny  ,nz  )
    τyy        = @zeros(nx  ,ny  ,nz  )
    τzz        = @zeros(nx  ,ny  ,nz  )
    τxy        = @zeros(nx-1,ny-1,nz-2)
    τxz        = @zeros(nx-1,ny-2,nz-1)
    τyz        = @zeros(nx-2,ny-1,nz-1)
    εxx        = @zeros(nx  ,ny  ,nz  )
    εyy        = @zeros(nx  ,ny  ,nz  )
    εzz        = @zeros(nx  ,ny  ,nz  )
    εxy        = @zeros(nx-1,ny-1,nz-2)
    εxz        = @zeros(nx-1,ny-2,nz-1)
    εyz        = @zeros(nx-2,ny-1,nz-1)
    εII        = @zeros(nx-2,ny-2,nz-2)
    η          = @zeros(nx  ,ny  ,nz  )
    Vmag       = @zeros(nx  ,ny  ,nz  )
    dPr        = @zeros(nx  ,ny  ,nz  )
    r_Vx       = @zeros(nx-1,ny-2,nz-2)
    r_Vy       = @zeros(nx-2,ny-1,nz-2)
    r_Vz       = @zeros(nx-2,ny-2,nz-1)
    ρgz_c      = @zeros(nx  ,ny  ,nz  )
    ρgx        = @zeros(nx-1,ny-2,nz-2)
    ρgy        = @zeros(nx-2,ny-1,nz-2)
    ρgz        = @zeros(nx-2,ny-2,nz-1)
    ph_ice     = @zeros(nx  ,ny  ,nz  )
    ph_bed     = @zeros(nx  ,ny  ,nz  )
    ητ         = @zeros(nx  ,ny  ,nz  )
    U          = @zeros(nx  ,ny  ,nz  )
    T          = @zeros(nx  ,ny  ,nz  )
    qUx        = @zeros(nx-1,ny-2,nz-2)
    qUy        = @zeros(nx-2,ny-1,nz-2)
    qUz        = @zeros(nx-2,ny-2,nz-1)
    # initialisation
    @parallel (1:nx,1:ny,1:nz) compute_phase!(ρgz_c,ph_ice,ph_bed,xc,yc,zc,r_box,w_box,r_rnd,z_bed,δ_sd,ρg0.air,ρg0.ice,ρg0.bed)
    @parallel init_U!(U,T,ρCp.ice,T0.ice)
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
            @parallel (1:size(η,1),1:size(η,2),1:size(η,3)) compute_η!(η,T,Q_R,η0.air,η0.ice,η0.bed,ph_ice,ph_bed)
            @parallel update_iter_params!(ητ,η)
            @parallel (1:size(ητ,2),1:size(ητ,3)) bc_x!(ητ)
            @parallel (1:size(ητ,1),1:size(ητ,3)) bc_y!(ητ)
            @parallel (1:size(ητ,1),1:size(ητ,2)) bc_z!(ητ)
            @parallel update_normal_τ!(Pr,dPr,εxx,εyy,εzz,εxy,εxz,εyz,Vx,Vy,Vz,∇V,η,r,θ_dτ,dx,dy,dz)
            @parallel update_shear_τ!(τxx,τyy,τzz,τxy,τxz,τyz,εxx,εyy,εzz,εxy,εxz,εyz,εII,η,dτ_r)
            @parallel update_velocities!(Vx,Vy,Vz,Pr,τxx,τyy,τzz,τxy,τxz,τyz,ητ,ρgx,ρgy,ρgz,nudτ,dx,dy,dz)
            # free slip x
            @parallel (1:size(Vx,1),1:size(Vx,3)) bc_y!(Vx)
            @parallel (1:size(Vx,1),1:size(Vx,2)) bc_z!(Vx)
            # free slip y
            @parallel (1:size(Vy,2),1:size(Vy,3)) bc_x!(Vy)
            @parallel (1:size(Vy,1),1:size(Vy,2)) bc_z!(Vy)
            # free slip z
            @parallel (1:size(Vz,2),1:size(Vz,3)) bc_x!(Vz)
            @parallel (1:size(Vz,1),1:size(Vz,3)) bc_y!(Vz)
            # free surface top
            @parallel (1:size(Vx,2),1:size(Vx,3)) bc_Vx!(Vx,η,Pr,ph_ice,ph_bed,dx)
            @parallel (1:size(Vy,1),1:size(Vy,3)) bc_Vy!(Vy,η,Pr,ph_ice,ph_bed,dy)
            @parallel (1:size(Vz,1),1:size(Vz,2)) bc_Vz!(Vz,η,Pr,dz)
            # no slip bottom
            @parallel (1:size(Vx,1),1:size(Vx,2)) bc_Vxy!(Vx)
            @parallel (1:size(Vy,1),1:size(Vy,2)) bc_Vxy!(Vy)
            if iter % ncheck == 0
                @parallel compute_residuals!(r_Vx,r_Vy,r_Vz,Pr,τxx,τyy,τzz,τxy,τxz,τyz,ρgx,ρgy,ρgz,dx,dy,dz)
                errs = maximum.((abs.(r_Vx),abs.(r_Vy),abs.(r_Vz),abs.(dPr[2:end-1,2:end-1,2:end-1])))
                push!(iter_evo,iter/min(nx,ny,nz));append!(errs_evo,errs)
                @printf("  iter/nz=%.3f,errs=[ %1.3e, %1.3e, %1.3e, %1.3e ] \n",iter/min(nx,ny,nz),errs...)
            end
            iter += 1
        end
        # thermal
        @parallel compute_T!(T,U,ρCp.ice)
        @parallel update_qU!(qUx,qUy,qUz,T,λ.ice,dx,dy,dz)
        @parallel update_U!(U,qUx,qUy,qUz,η,εII,dt,dx,dy,dz)
        @parallel (1:size(U,2),1:size(U,3)) bc_x!(U)
        @parallel (1:size(U,1),1:size(U,3)) bc_y!(U)
        @parallel (1:size(U,1),1:size(U,2)) bc_z!(U)
    end
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
    return
end

main()
