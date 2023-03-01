using FastIce
using MPI
using ImplicitGlobalGrid
using TinyKernels

using GLMakie

include("load_dem.jl")
include("signed_distances.jl")
include("level_sets.jl")

@views av1(A) = 0.5.*(A[1:end-1].+A[2:end])

@views function main(grid_dims)
    # path to DEM data
    greenland_path = "data/BedMachine/greenland.jld2"

    # region to simulate
    global_region = (xlims = (1100.0e3,1200.0e3), ylims = (1000.0e3,1100.0e3))

    # load DEM
    @info "loading DEM data from the file '$greenland_path'"
    (;x,y,bed,surface) = load_dem(greenland_path,global_region)

    # TODO: remove
    # exagerrating the elevation
    bed     .*= 2
    surface .*= 2

    # compute origin and size of the domain (required for scaling and computing the grid size)
    ox,oy,oz = x[1], y[1], minimum(bed)
    lx = x[end] - ox
    ly = y[end] - oy
    lz = maximum(surface) - oz

    # shift and scale the domain before computation (center of the domain is (0,0) in x-y plane)
    δx, δy = ox + 0.5lx,oy + 0.5ly # required to avoid conversion to Vector  
    x = @. (x - δx)/lz
    y = @. (y - δy)/lz
    @. bed     = (bed     - oz)/lz
    @. surface = (surface - oz)/lz

    # plot the selected region
    fig = Figure(resolution=(1500,700),fontsize=32)
    axs = (
        bed     = Axis(fig[1,1][1,1];aspect=DataAspect(),title="bed"),
        surface = Axis(fig[1,2][1,1];aspect=DataAspect(),title="surface"),
    )
    plts = (
        bed     = heatmap!(axs.bed    ,x,y,bed    ),
        surface = heatmap!(axs.surface,x,y,surface),
    )
    Colorbar(fig[1,1][1,2],plts.bed)
    Colorbar(fig[1,2][1,2],plts.surface)
    display(fig)

    # run simulation
    dem_data = (;x,y,bed,surface)
    @info "running the simulation"
    run_simulation(dem_data,grid_dims)
    
    return
end

@views function run_simulation(dem_data,grid_dims)
    # init MPI
    MPI.Initialized() || MPI.Init()
    me, dims, nprocs, coords, comm_cart = init_global_grid(grid_dims...;init_MPI=false)
    dims   = Tuple(dims)
    coords = Tuple(coords)

    # physics
    # global domain origin and size
    ox_g, oy_g, oz_g = dem_data.x[1], dem_data.y[1], 0.0
    lx_g = dem_data.x[end] - ox_g
    ly_g = dem_data.y[end] - oy_g
    lz_g = 1.0

    # local domain size and origin
    lx_l,ly_l,lz_l = (lx_g,ly_g,lz_g)./dims
    ox_l,oy_l,oz_l = (ox_g,oy_g,oz_g) .+ coords.*(lx_l,ly_l,lz_l)

    # numerics
    nx,ny,nz = grid_dims
    dx,dy,dz = lx_g/nx_g(), ly_g/ny_g(), lz_g/nz_g()

    # preprocessing
    xv_l = LinRange(ox_l,ox_l+lx_l,nx+1)
    yv_l = LinRange(oy_l,oy_l+ly_l,ny+1)
    zv_l = LinRange(oz_l,oz_l+lz_l,nz+1)
    xc_l,yc_l,zc_l = av1.((xv_l,yv_l,zv_l))

    # fields allocation
    # level set
    Ψ = (
        fluid  = field_array(Float64,nx+1,ny+1,nz+1),
        liquid = field_array(Float64,nx+1,ny+1,nz+1),
    )
    # wt = (
    #     c = (
    #         fluid  = field_array(Float64,nx,ny,nz),
    #         liquid = field_array(Float64,nx,ny,nz),
    #     ),
    #     x = (
    #         fluid  = field_array(Float64,nx+1,ny,nz),
    #         liquid = field_array(Float64,nx+1,ny,nz),
    #     ),
    #     y = (
    #         fluid  = field_array(Float64,nx,ny+1,nz),
    #         liquid = field_array(Float64,nx,ny+1,nz),
    #     ),
    #     z = (
    #         fluid  = field_array(Float64,nx,ny,nz+1),
    #         liquid = field_array(Float64,nx,ny,nz+1),
    #     ),
    #     xy = (
    #         fluid  = field_array(Float64,nx-1,ny-1,nz-2),
    #         liquid = field_array(Float64,nx-1,ny-1,nz-2),
    #     ),
    #     xz = (
    #         fluid  = field_array(Float64,nx-1,ny-2,nz-1),
    #         liquid = field_array(Float64,nx-1,ny-2,nz-1),
    #     ),
    #     yz = (
    #         fluid  = field_array(Float64,nx-2,ny-1,nz-1),
    #         liquid = field_array(Float64,nx-2,ny-1,nz-1),
    #     ),
    # )
    # mechanics
    # Pr = field_array(Float64,nx,ny,nz)
    # τ  = (
    #     xx = field_array(Float64,nx  ,ny  ,nz  ),
    #     yy = field_array(Float64,nx  ,ny  ,nz  ),
    #     zz = field_array(Float64,nx  ,ny  ,nz  ),
    #     xy = field_array(Float64,nx-1,ny-1,nz-2),
    #     xz = field_array(Float64,nx-1,ny-2,nz-1),
    #     yz = field_array(Float64,nx-2,ny-1,nz-1),
    # )
    # V = (
    #     x = field_array(Float64,nx+1,ny,nz),
    #     y = field_array(Float64,nx,ny+1,nz),
    #     z = field_array(Float64,nx,ny,nz+1),
    # )
    # residuals
    # Res = (
    #     Pr = field_array(Float64,nx,ny,nz),
    #     V = (
    #         x = field_array(Float64,nx-1,ny-2,nz-2),
    #         y = field_array(Float64,nx-2,ny-1,nz-2),
    #         z = field_array(Float64,nx-2,ny-2,nz-1),
    #     )
    # )

    # figures
    fig = Figure(resolution=(1500,700),fontsize=32)
    axs = (
        dem = (
            bed     = Axis3(fig[1,1][1,1];title="bed"    ,aspect=:data),
            surface = Axis3(fig[1,2][1,1];title="surface",aspect=:data),
        ),
        Ψ = (
            fluid  = Axis3(fig[2,1][1,1];title="bed"    ,aspect=:data),
            liquid = Axis3(fig[2,2][1,1];title="surface",aspect=:data),
        ),
    )
    plts = (
        dem = (
            bed     = surface!(axs.dem.bed    ,dem_data.x,dem_data.y,dem_data.bed    ),
            surface = surface!(axs.dem.surface,dem_data.x,dem_data.y,dem_data.surface),
        ),
        Ψ = (
            fluid  = volume!(axs.Ψ.fluid ,xv_l,yv_l,zv_l,Array(Ψ.fluid) ;algorithm=:iso,isovalue=0.0),
            liquid = volume!(axs.Ψ.liquid,xv_l,yv_l,zv_l,Array(Ψ.liquid);algorithm=:iso,isovalue=0.0),
        ),
    )
    # Colorbar(fig[1,1][1,2],plts.Ψ.fluid )
    # Colorbar(fig[1,2][1,2],plts.Ψ.liquid)

    # initialisation
    # for comp in eachindex(V) fill!(V[comp],0.0) end
    # for comp in eachindex(τ) fill!(τ[comp],0.0) end
    # fill!(Pr,0.0)

    # compute level sets from DEM data
    dem_grid = (dem_data.x,dem_data.y)
    Ψ_grid   = (xv_l,yv_l,zv_l)
    
    @info "computing the level set for the ice surface"
    compute_level_set_from_dem!(Ψ.liquid,to_device(dem_data.surface),dem_grid,Ψ_grid)
    
    @info "computing the level set for the bedrock surface"
    compute_level_set_from_dem!(Ψ.fluid,to_device(dem_data.bed),dem_grid,Ψ_grid)

    # update plots
    plts.Ψ.fluid[4]  = Array(Ψ.fluid)
    plts.Ψ.liquid[4] = Array(Ψ.liquid)
    display(fig)

    finalize_global_grid(;finalize_MPI=false)
    # MPI.Finalize()
    return
end

main((128,128,32))