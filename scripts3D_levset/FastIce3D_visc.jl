const USE_GPU = haskey(ENV,"USE_GPU") ? parse(Bool,ENV["USE_GPU"]) : true
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA,Float64,3)
else
    @init_parallel_stencil(Threads,Float64,3)
end
using ElasticArrays,Printf
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
macro d_xii(A) esc(:( $A[$ixi+1,$iyi  ,$izi  ]-$A[$ixi,$iyi,$izi] )) end
macro d_yii(A) esc(:( $A[$ixi  ,$iyi+1,$izi  ]-$A[$ixi,$iyi,$izi] )) end
macro d_zii(A) esc(:( $A[$ixi  ,$iyi  ,$izi+1]-$A[$ixi,$iyi,$izi] )) end

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

macro d_x_iy(A::Symbol)  esc(:( $A[$ix+1,$iyi,$iz ] - $A[$ix ,$iyi,$iz ] )) end
macro d_x_iz(A::Symbol)  esc(:( $A[$ix+1,$iy ,$izi] - $A[$ix ,$iy ,$izi] )) end

macro d_y_ix(A::Symbol)  esc(:( $A[$ixi,$iy+1,$iz ] - $A[$ixi,$iy ,$iz ] )) end
macro d_y_iz(A::Symbol)  esc(:( $A[$ix ,$iy+1,$izi] - $A[$ix ,$iy ,$izi] )) end

macro d_z_ix(A::Symbol)  esc(:( $A[$ixi,$iy ,$iz+1] - $A[$ixi,$iy ,$iz ] )) end
macro d_z_iy(A::Symbol)  esc(:( $A[$ix ,$iyi,$iz+1] - $A[$ix ,$iyi,$iz ] )) end

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
    Vz[ix,iy,end] = Vz[ix,iy,end-1] + 0.5*dz/η[ix,iy,end]*(Pr[ix,iy,end] + 1.0/3.0*(-Pr[ix,iy,end-1] + 2.0*η[ix,iy,end-1]*(Vz[ix,iy,end-1] - Vz[ix,iy,end-2])/dz))
    return
end

@parallel function compute_residuals!(r_Vx,r_Vy,r_Vz,Pr,τxx,τyy,τzz,τxy,τxz,τyz,ρgx,ρgy,ρgz,dx,dy,dz)
    @all(r_Vx) = -@d_xi(Pr)/dx + @d_xi(τxx)/dx + @d_ya(τxy)/dy + @d_za(τxz)/dz - @all(ρgx)
    @all(r_Vy) = -@d_yi(Pr)/dy + @d_yi(τyy)/dy + @d_xa(τxy)/dx + @d_za(τyz)/dz - @all(ρgy)
    @all(r_Vz) = -@d_zi(Pr)/dz + @d_zi(τzz)/dz + @d_xa(τxz)/dx + @d_ya(τyz)/dy - @all(ρgz)
    return
end

@parallel_indices (ix,iy,iz) function compte_η_ρg!(η,ρgz_c,phase,xc,yc,zc,x0,y0,z0,r_dep,δ_sd,η0_air,η0_ice,ρg0_air,ρg0_ice)
    if ix<=size(η,1) && iy<=size(η,2) && iz<=size(η,3)
        sd_air = sqrt((xc[ix]-x0)^2 + (yc[iy]-y0)^2+(zc[iz]-z0)^2)-r_dep
        t_air  = 0.5*(tanh(-sd_air/δ_sd) + 1)
        t_ice  = 1.0 - t_air
        η[ix,iy,iz]     = t_ice*η0_ice  + t_air*η0_air
        ρgz_c[ix,iy,iz] = t_ice*ρg0_ice + t_air*ρg0_air
        phase[ix,iy,iz] =  1.0 - t_air
    end
    return
end

@parallel function init_U!(U,T,ρCp,T0)
    @all(T) = T0
    @all(U) = ρCp*@all(T)
    return
end

@views function main()
    # physics
    lx,ly,lz   = 20.0,20.0,10.0
    η0         = (ice = 1.0 , air = 1e-4)
    ρg0        = (ice = 1.0 , air = 0.0 )
    λ          = (ice = 1.0 , air = 1.0 )
    ρCp        = (ice = 1.0 , air = 1.0 )
    T0         = (ice = 253.0,air = 253.0)
    r_dep      = 3.0*min(lx,ly,lz)
    x0,y0,z0   = 0.1lx,0.2ly,0.8lz + sqrt(r_dep^2-max(lx,ly)^2/4.0)
    # numerics
    nx         = 64
    ny         = ceil(Int,nx*ly/lx)
    nz         = ceil(Int,nx*lz/lx)
    ϵtol       = (1e-6,1e-6,1e-6,1e-6)
    maxiter    = 20max(nx,ny,nz)
    ncheck     = ceil(Int,2max(nx,ny,nz))
    r          = 0.5
    re_mech    = 2π
    nt         = 10
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
    phase      = @zeros(nx  ,ny  ,nz  )
    ητ         = @zeros(nx  ,ny  ,nz  )
    U          = @zeros(nx  ,ny  ,nz  )
    T          = @zeros(nx  ,ny  ,nz  )
    qUx        = @zeros(nx-1,ny-2,nz-2)
    qUy        = @zeros(nx-2,ny-1,nz-2)
    qUz        = @zeros(nx-2,ny-2,nz-1)
    # initialisation
    @parallel compte_η_ρg!(η,ρgz_c,phase,xc,yc,zc,x0,y0,z0,r_dep,δ_sd,η0.air,η0.ice,ρg0.air,ρg0.ice)
    @parallel init_U!(U,T,ρCp.ice,T0.ice)
    ρgz .= ameanz(ρgz_c[2:end-1,2:end-1,:])
    Pr  .= Data.Array(reverse(cumsum(reverse(Array(ρgz_c),dims=3),dims=3).*dz,dims=3))
    iter_evo=Float64[]; errs_evo=ElasticMatrix{Float64}(undef,length(ϵtol),0)
    # time loop
    for it = 1:nt
        @printf("it = %d \n",it)
        errs = 2.0.*ϵtol; iter = 1
        resize!(iter_evo,0); resize!(errs_evo,length(ϵtol),0)
        # iteration loop
        while any(errs .>= ϵtol) && iter <= maxiter
            # mechanics
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
            # no slip bottom
            @parallel (1:size(Vx,1),1:size(Vx,2)) bc_Vxy!(Vx)
            @parallel (1:size(Vy,1),1:size(Vy,2)) bc_Vxy!(Vy)
            # free surface top
            @parallel (1:size(Vz,1),1:size(Vz,2)) bc_Vz!(Vz,η,Pr,dz)
            if iter % ncheck == 0
                @parallel compute_residuals!(r_Vx,r_Vy,r_Vz,Pr,τxx,τyy,τzz,τxy,τxz,τyz,ρgx,ρgy,ρgz,dx,dy,dz)
                errs = maximum.((abs.(r_Vx),abs.(r_Vy),abs.(r_Vz),abs.(dPr)))
                push!(iter_evo,iter/max(nx,ny));append!(errs_evo,errs)
                @printf("  iter/nx=%.3f,errs=[ %1.3e, %1.3e, %1.3e, %1.3e ] \n",iter/max(nx,ny),errs...)
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
    mask = copy(phase); @. mask[mask<0.7]=NaN
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
