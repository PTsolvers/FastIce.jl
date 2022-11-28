using FastIce
using FastIce.LevelSets.AMDGPUBackend

using GLMakie,HDF5,AMDGPU,LinearAlgebra,Printf

function make_levelsets!(bed_ls,ice_ls,mask,dldt,bed_dem,surf_dem,rc,dem_rc,dx,dy,dz,cutoff,R)
    init_level_set!(bed_ls,mask,bed_dem,rc,dem_rc,cutoff,R)
    solve_eikonal!(bed_ls,dldt,mask,dx,dy,dz)
    init_level_set!(ice_ls,mask,surf_dem,rc,dem_rc,cutoff,R)
    solve_eikonal!(ice_ls,dldt,mask,dx,dy,dz)
    return
end

function init_timing(dem_filename,nz)
    t_tic = time_ns()
    io  = h5open(dem_filename,"r")
    bed = CuArray(read(io,"bed"))
    ice = CuArray(read(io,"ice"))
    @read io lx; @read io ly
    @read io ox; @read io oy
    close(io)
    dem_nx,dem_ny = size(bed)
    dem_dx,dem_dy = lx/dem_nx,ly/dem_ny
    dem_xc,dem_yc = ox .+ LinRange(dem_dx/2,lx-dem_dx/2,dem_nx),oy .+ LinRange(dem_dy/2,ly-dem_dy/2,dem_ny)
    surf = bed .+ ice
    oz,zmax = extrema(surf)
    lz = zmax - oz
    oz -= 0.1lz
    lz *= 1.2
    nx = ceil(Int,nz*lx/lz)
    ny = ceil(Int,nz*ly/lz)
    dx,dy,dz = lx/nx,ly/ny,lz/nz
    xc,yc,zc = ox .+ LinRange(dx/2,lx-dx/2,nx), oy .+ LinRange(dy/2,ly-dy/2,ny), oz .+ LinRange(dz/2,lz-dz/2,nz)
    # array allocation
    bed_ls = RocArray{Float64}(undef,nx,ny,nz)
    ice_ls = RocArray{Float64}(undef,nx,ny,nz)
    dldt   = RocArray{Float64}(undef,nx,ny,nz)
    mask   = RocArray{Bool}(undef,nx,ny,nz)
    # make levelsets
    cutoff = sqrt(dx^2+dy^2+dz^2)
    make_levelsets!(bed_ls,ice_ls,mask,dldt,bed,surf,(xc,yc,zc),(dem_xc,dem_yc),dx,dy,dz,cutoff,I)
    @. ice_ls = max(ice_ls,-bed_ls)
    t_tot = (time_ns() - t_tic)*1e-9
    @printf("elapsed time: %.2f s\n",t_tot)
    # visualisation
    fig = Figure(fontsize=32,resolution=(2000,1600))
    ax1 = Axis3(fig[1,1];aspect=:data)
    ax2 = Axis3(fig[2,1];aspect=:data)
    surface!(ax1,dem_xc,dem_yc,Array(bed );colormap=:turbo)
    surface!(ax1,dem_xc,dem_yc,Array(surf);colormap=:winter)
    volume!(ax2,xc,yc,zc,Array(bed_ls);algorithm=:iso,isovalue=0.0,isorange=0.5,colormap=:turbo )
    volume!(ax2,xc,yc,zc,Array(ice_ls);algorithm=:iso,isovalue=0.0,isorange=0.5,colormap=:winter)
    display(fig)
    return
end
