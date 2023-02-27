using FastIce
using FastIce.LevelSets.CUDABackend

using NetCDF

using GLMakie,Colors
using CUDA,LinearAlgebra,Printf

function make_levelsets!(bed_ls,ice_ls,mask,dldt,bed_dem,surf_dem,rc,dem_rc,dx,dy,dz,cutoff,R)
    init_level_set!(bed_ls,mask,bed_dem,rc,dem_rc,cutoff,R)
    solve_eikonal!(bed_ls,dldt,mask,dx,dy,dz)
    init_level_set!(ice_ls,mask,surf_dem,rc,dem_rc,cutoff,R)
    solve_eikonal!(ice_ls,dldt,mask,dx,dy,dz)
    return
end

function main()
    filename = "..\\..\\Datasets\\BedMachineGreenland\\BedMachineGreenland-v5.nc"
    # ix_rng = 1:100:10000
    # iy_rng = 1:100:18000
    ix_rng = 7800:8500
    iy_rng = 10300:11300
    bed_dem = ncread(filename,"bed")[ix_rng,iy_rng]
    ice_dem = ncread(filename,"surface")[ix_rng,iy_rng]
    x_dem   = ncread(filename,"x")[ix_rng]
    y_dem   = ncread(filename,"y")[iy_rng]

    ox,oy = x_dem[1],y_dem[1]

    reverse!(bed_dem;dims=2)
    reverse!(ice_dem;dims=2)
    reverse!(y_dem)

    x_dem .-= minimum(x_dem)
    y_dem .-= minimum(y_dem)

    xc_dem = LinRange(x_dem[1],x_dem[end],length(x_dem))
    yc_dem = LinRange(y_dem[1],y_dem[end],length(y_dem))

    min_bed = minimum(bed_dem)
    bed_dem  .-= min_bed
    ice_dem .-= min_bed
    bed_dem .*= 2.0; ice_dem .*= 2.0 # exagerrate

    lz     = maximum(ice_dem)

    oz = -0.15lz
    lz *= 1.3

    lx,ly  = xc_dem[end]-xc_dem[1],yc_dem[end]-yc_dem[1]
    nz     = 32
    nx,ny  = ceil(Int,nz*lx/lz),ceil(Int,nz*ly/lz)
    dx,dy,dz = lx/(nx-1),ly/(ny-1),lz/(nz-1)
    xc,yc,zc = LinRange(xc_dem[1],xc_dem[end],nx),LinRange(yc_dem[1],yc_dem[end],ny),LinRange(oz,oz+lz,nz)

    @show lx ly lz
    @show nx ny nz

    bed_ls = CuArray{Float64}(undef,nx,ny,nz)
    ice_ls = CuArray{Float64}(undef,nx,ny,nz)
    dldt   = CuArray{Float64}(undef,nx,ny,nz)
    mask   = CuArray{Bool}(undef,nx,ny,nz)
    cutoff = sqrt(dx^2+dy^2+dz^2)
    rc     = (xc,yc,zc)
    rc_dem = (xc_dem,yc_dem)
    R      = LinearAlgebra.I

    init_level_set!(bed_ls,mask,CuArray(bed_dem),rc,rc_dem,cutoff,R)
    solve_eikonal!(bed_ls,dldt,mask,dx,dy,dz)
    init_level_set!(ice_ls,mask,CuArray(ice_dem),rc,rc_dem,cutoff,R)
    solve_eikonal!(ice_ls,dldt,mask,dx,dy,dz)

    # @. ice_ls = max(ice_ls,-(bed_ls.-200.0))

    fig = Figure(resolution=(2600,1600),fontsize=38,backgroundcolor=RGB(242/255,242/255,242/255))
    ax1  = Axis3(fig[1,2][1,1];aspect=:data,title="Mesh representation",
        xticklabelsvisible=false,
        yticklabelsvisible=false,
        zticklabelsvisible=false,
        backgroundcolor=:transparent)
    xlims!(ax1, xc[1]+200, xc[end]-200)

    ax2  = Axis3(fig[1,2][2,1];aspect=:data,title="Level set representation",
        xticklabelsvisible=false,
        yticklabelsvisible=false,
        zticklabelsvisible=false,
        backgroundcolor=:transparent)

    surface!(ax1,xc_dem,yc_dem,ice_dem;colormap=:terrain)
    ice_color = fill(RGBA(0.8,1.0,1.0,0.4),size(ice_dem)...)
    @. ice_color[ice_dem<=bed_dem] = RGBA(0.8,1.0,1.0,0.0)
    # surface!(ax1,xc_dem,yc_dem,ice_dem;color=ice_color)
    st = 30
    wireframe!(ax1,xc_dem[1:st:end],yc_dem[1:st:end],ice_dem[1:st:end,1:st:end].+250;linewidth=3,color=RGBA(0.0,0.0,0.0,0.3))
    
    # volume!(ax2,xc,yc,zc,Array(bed_ls);algorithm=:iso,isovalue=0.0,isorange=100,colormap=:summer)
    # volume!(ax2,xc,yc,zc,Array(ice_ls);algorithm=:iso,isovalue=0.0,isorange=100,colormap=:BuPu_3)

    plt = volumeslices!(ax2,xc,yc,zc,Array(ice_ls);colormap=:turbo)
    xlims!(ax2,xc[1],xc[end])
    ylims!(ax2,yc[1],yc[end])
    plt[:update_yz][](round(Int,0.75length(xc)))
    plt[:update_xz][](round(Int,0.75length(yc)))
    plt[:update_xy][](round(Int,0.05length(zc)))

    ax3 = Axis(fig[1,1][2,1];aspect=DataAspect(),title="Greenland DEM regional",
    xticklabelsvisible=false,yticklabelsvisible=false,backgroundcolor=:transparent)
    colsize!(fig.layout, 1, Relative(1/3))
    hm = heatmap!(ax3,xc_dem,yc_dem,ice_dem;colormap=:terrain)
    # Colorbar(fig[1,1][1,1],hm)
    
    ix_full = 1:50:10000
    iy_full = 1:50:18000

    ice_full  = ncread(filename,"surface")[ix_full,iy_full]
    mask_full = ncread(filename,"mask")[ix_full,iy_full]
    x_full   = ncread(filename,"x")[ix_full]
    y_full   = ncread(filename,"y")[iy_full]

    reverse!(ice_full;dims=2)
    reverse!(mask_full;dims=2)
    reverse!(y_full)

    @. ice_full[mask_full == 0] = NaN

    ax4 = Axis(fig[1,1][1,1];aspect=DataAspect(),title="Greenland DEM full",
    xticklabelsvisible=false,yticklabelsvisible=false)
    colsize!(fig.layout, 1, Relative(1/3))
    hm = heatmap!(ax4,x_full,y_full,ice_full;colormap=:terrain)

    poly!(ax4, Point2f[(ox+xc_dem[1]  , oy+yc_dem[1]),
                       (ox+xc_dem[end], oy+yc_dem[1]),
                       (ox+xc_dem[end], oy+yc_dem[end]),
                       (ox+xc_dem[1], oy+yc_dem[end])];
                    strokecolor = :red, strokewidth = 3, color=RGBA(0.5,0.4,0.4,0.6))
    # Colorbar(fig[1,1][1,1],hm)
    
    # display(fig)

    save("greenland.png",fig)
    return
end

main()