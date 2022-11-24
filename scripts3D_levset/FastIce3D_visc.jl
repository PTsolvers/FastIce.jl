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
    @inn(εxy) = 0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx)
    @inn(εxz) = 0.5*(@d_zi(Vx)/dz + @d_xi(Vz)/dx)
    @inn(εyz) = 0.5*(@d_zi(Vy)/dz + @d_yi(Vz)/dy)
    return
end

@parallel function update_shear_τ!(τxx,τyy,τzz,τxy,τxz,τyz,εxx,εyy,εzz,εxy,εxz,εyz,η,dτ_r)
    @all(τxx)  = @all(τxx) + (-@all(τxx) + 2.0*@all(η)*@all(εxx))*dτ_r
    @all(τyy)  = @all(τyy) + (-@all(τyy) + 2.0*@all(η)*@all(εyy))*dτ_r
    @all(τzz)  = @all(τzz) + (-@all(τzz) + 2.0*@all(η)*@all(εzz))*dτ_r
    @inn_xy(τxy) = @inn_xy(τxy) + (-@inn_xy(τxy) + 2.0*@av_xya(η)*@inn_xy(εxy))*dτ_r
    @inn_xz(τxz) = @inn_xz(τxz) + (-@inn_xz(τxz) + 2.0*@av_xza(η)*@inn_xz(εxz))*dτ_r
    @inn_yz(τyz) = @inn_yz(τyz) + (-@inn_yz(τyz) + 2.0*@av_yza(η)*@inn_yz(εyz))*dτ_r
    return
end

macro d_x_iy(A::Symbol)  esc(:( $A[$ix+1,$iyi,$iz ] - $A[$ix ,$iyi,$iz ] )) end
macro d_x_iz(A::Symbol)  esc(:( $A[$ix+1,$iy ,$izi] - $A[$ix ,$iy ,$izi] )) end

macro d_y_ix(A::Symbol)  esc(:( $A[$ixi,$iy+1,$iz ] - $A[$ixi,$iy ,$iz ] )) end
macro d_y_iz(A::Symbol)  esc(:( $A[$ix ,$iy+1,$izi] - $A[$ix ,$iy ,$izi] )) end

macro d_z_ix(A::Symbol)  esc(:( $A[$ixi,$iy ,$iz+1] - $A[$ixi,$iy ,$iz ] )) end
macro d_z_iy(A::Symbol)  esc(:( $A[$ix ,$iyi,$iz+1] - $A[$ix ,$iyi,$iz ] )) end

@parallel function update_velocities!(Vx,Vy,Vz,Pr,τxx,τyy,τzz,τxy,τxz,τyz,ητ,ρgx,ρgy,ρgz,nudτ,dx,dy,dz)
    @inn_x(Vx) = @inn_x(Vx) + (-@d_xa(Pr)/dx + @d_xa(τxx)/dx + @d_y_ix(τxy)/dy + @d_z_ix(τxz)/dz - @all(ρgx))*nudτ/@av_xa(ητ)
    @inn_y(Vy) = @inn_y(Vy) + (-@d_ya(Pr)/dy + @d_ya(τyy)/dy + @d_x_iy(τxy)/dx + @d_z_iy(τyz)/dz - @all(ρgy))*nudτ/@av_ya(ητ)
    @inn_z(Vz) = @inn_z(Vz) + (-@d_za(Pr)/dz + @d_za(τzz)/dz + @d_x_iz(τxz)/dx + @d_y_iz(τyz)/dy - @all(ρgz))*nudτ/@av_za(ητ)
    return
end

@parallel_indices (ix,iy) function bc_Vxy!(Vxy)
    Vxy[ix,iy,1] = 1.0/3.0*Vxy[ix,iy,2]
    return
end

@parallel_indices (ix,iy) function bc_Vz!(Vz,η,Pr,dz)
    Vz[ix,iy,end] = Vz[ix,iy,end-1] + 0.5*dz/η[ix,iy,end]*(Pr[ix,iy,end] + 1.0/3.0*(-Pr[ix,iy,end-1] + 2.0*η[ix,iy,end-1]*(Vz[ix,iy,end-1] - Vz[ix,iy,end-2])/dz))
    return
end

@parallel function compute_residuals!(r_Vx,r_Vy,r_Vz,Pr,τxx,τyy,τzz,τxy,τxz,τyz,ρgx,ρgy,ρgz,dx,dy,dz)
    @all(r_Vx) = -@d_xi(Pr)/dx + @d_xi(τxx)/dx + @d_yii(τxy)/dy + @d_zii(τxz)/dz - @inn_yz(ρgx)
    @all(r_Vy) = -@d_yi(Pr)/dy + @d_yi(τyy)/dy + @d_xii(τxy)/dx + @d_zii(τyz)/dz - @inn_xz(ρgy)
    @all(r_Vz) = -@d_zi(Pr)/dz + @d_zi(τzz)/dz + @d_xii(τxz)/dx + @d_yii(τyz)/dy - @inn_xy(ρgz)
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

@views function main()
    # physics
    lx,ly,lz   = 20.0,20.0,10.0
    η0         = (ice = 1.0 , air = 1e-6)
    ρg0        = (ice = 1.0 , air = 0.0 )
    r_dep      = 3.0*min(lx,ly,lz)
    x0,y0,z0   = 0.1lx,0.2ly,0.8lz + sqrt(r_dep^2-max(lx,ly)^2/4.0)
    # numerics
    nx         = 128
    ny         = ceil(Int,nx*ly/lx)
    nz         = ceil(Int,nx*lz/lx)
    ϵtol       = (1e-6,1e-6,1e-6,1e-6)
    maxiter    = 20max(nx,ny,nz)
    ncheck     = ceil(Int,2max(nx,ny,nz))
    r          = 0.5
    re_mech    = 2π
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
    # array allocation
    Vx         = @zeros(nx+1,ny  ,nz  )
    Vy         = @zeros(nx  ,ny+1,nz  )
    Vz         = @zeros(nx  ,ny  ,nz+1)
    Pr         = @zeros(nx  ,ny  ,nz  )
    ∇V         = @zeros(nx  ,ny  ,nz  )
    τxx        = @zeros(nx  ,ny  ,nz  )
    τyy        = @zeros(nx  ,ny  ,nz  )
    τzz        = @zeros(nx  ,ny  ,nz  )
    τxy        = @zeros(nx+1,ny+1,nz  )
    τxz        = @zeros(nx+1,ny  ,nz+1)
    τyz        = @zeros(nx  ,ny+1,nz+1)
    εxx        = @zeros(nx  ,ny  ,nz  )
    εyy        = @zeros(nx  ,ny  ,nz  )
    εzz        = @zeros(nx  ,ny  ,nz  )
    εxy        = @zeros(nx+1,ny+1,nz  )
    εxz        = @zeros(nx+1,ny  ,nz+1)
    εyz        = @zeros(nx  ,ny+1,nz+1)
    η          = @zeros(nx  ,ny  ,nz  )
    τII        = @zeros(nx  ,ny  ,nz  )
    εII        = @zeros(nx  ,ny  ,nz  )
    Vmag       = @zeros(nx  ,ny  ,nz  )
    dPr        = @zeros(nx  ,ny  ,nz  )
    r_Vx       = @zeros(nx-1,ny-2,nz-2)
    r_Vy       = @zeros(nx-2,ny-1,nz-2)
    r_Vz       = @zeros(nx-2,ny-2,nz-1)
    ρgz_c      = @zeros(nx  ,ny  ,nz  )
    ρgx        = @zeros(nx-1,ny  ,nz  )
    ρgy        = @zeros(nx  ,ny-1,nz  )
    ρgz        = @zeros(nx  ,ny  ,nz-1)
    phase      = @zeros(nx  ,ny  ,nz  )
    ητ         = @zeros(nx  ,ny  ,nz  )
    # initialisation
    @parallel compte_η_ρg!(η,ρgz_c,phase,xc,yc,zc,x0,y0,z0,r_dep,δ_sd,η0.air,η0.ice,ρg0.air,ρg0.ice)
    ρgz .= ameanz(ρgz_c)
    Pr  .= Data.Array(reverse(cumsum(reverse(Array(ρgz_c),dims=3),dims=3).*dz,dims=3))
    iter_evo=Float64[]; errs_evo=ElasticMatrix{Float64}(undef,length(ϵtol),0)
    errs = 2.0.*ϵtol; iter = 1
    resize!(iter_evo,0); resize!(errs_evo,length(ϵtol),0)
    # iteration loop
    while any(errs .>= ϵtol) && iter <= maxiter
        @parallel update_iter_params!(ητ,η)
        @parallel (1:size(ητ,2),1:size(ητ,3)) bc_x!(ητ)
        @parallel (1:size(ητ,1),1:size(ητ,3)) bc_y!(ητ)
        @parallel (1:size(ητ,1),1:size(ητ,2)) bc_z!(ητ)
        @parallel update_normal_τ!(Pr,dPr,εxx,εyy,εzz,εxy,εxz,εyz,Vx,Vy,Vz,∇V,η,r,θ_dτ,dx,dy,dz)
        @parallel update_shear_τ!(τxx,τyy,τzz,τxy,τxz,τyz,εxx,εyy,εzz,εxy,εxz,εyz,η,dτ_r)
        @parallel update_velocities!(Vx,Vy,Vz,Pr,τxx,τyy,τzz,τxy,τxz,τyz,ητ,ρgx,ρgy,ρgz,nudτ,dx,dy,dz)
        @parallel (1:size(Vx,1),1:size(Vx,2)) bc_Vxy!(Vx)
        @parallel (1:size(Vy,1),1:size(Vy,2)) bc_Vxy!(Vy)
        @parallel (1:size(Vz,1),1:size(Vz,2)) bc_Vz!(Vz,η,Pr,dz)
        if iter % ncheck == 0
            @parallel compute_residuals!(r_Vx,r_Vy,r_Vz,Pr,τxx,τyy,τzz,τxy,τxz,τyz,ρgx,ρgy,ρgz,dx,dy,dz)
            errs = maximum.((abs.(r_Vx),abs.(r_Vy),abs.(r_Vz),abs.(dPr)))
            push!(iter_evo,iter/max(nx,ny));append!(errs_evo,errs)
            @printf("  iter/nx=%.3f,errs=[ %1.3e, %1.3e, %1.3e, %1.3e ] \n",iter/max(nx,ny),errs...)
        end
        iter += 1
    end
    # visualisation
    Vmag .= sqrt.(ameanx(Vx).^2 .+ ameany(Vy).^2 .+ ameanz(Vz).^2)
    mask = copy(phase); @. mask[mask<0.7]=NaN
    fig = Figure(resolution=(3000,1000),fontsize=32)
    axs = (
        Pr   = Axis3(fig[1,1][1,1][1,1];aspect=:data,xlabel="x",ylabel="y",zlabel="z",title="Pr"),
        Vmag = Axis3(fig[1,1][1,2][1,1];aspect=:data,xlabel="x",ylabel="y",zlabel="z",title="|V|"),
    )
    plts = (
        Pr   = volumeslices!(axs.Pr  ,xc,yc,zc,Array(Pr  );colormap=:turbo),
        Vmag = volumeslices!(axs.Vmag,xc,yc,zc,Array(Vmag);colormap=:turbo),
    )
    sgrid = SliderGrid(
        fig[2,1],
        (label = "yz plane - x axis", range = 1:length(xc)),
        (label = "xz plane - y axis", range = 1:length(yc)),
        (label = "xy plane - z axis", range = 1:length(zc)),
    )
    # connect sliders to `volumeslices` update methods
    sl_yz, sl_xz, sl_xy = sgrid.sliders
    on(sl_yz.value) do v
        for prop in eachindex(plts) plts[prop][:update_yz][](v) end
    end
    on(sl_xz.value) do v
        for prop in eachindex(plts) plts[prop][:update_xz][](v) end
    end
    on(sl_xy.value) do v
        for prop in eachindex(plts) plts[prop][:update_xy][](v) end
    end
    set_close_to!(sl_yz, .5length(xc))
    set_close_to!(sl_xz, .5length(yc))
    set_close_to!(sl_xy, .5length(zc))
    [Colorbar(fig[1,1][irow,icol][1,2],plts[(irow-1)*2+icol]) for irow in 1:1,icol in 1:2]
    display(fig)
    return
end

main()
