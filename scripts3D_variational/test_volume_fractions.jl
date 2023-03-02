using FastIce
using MPI
using ImplicitGlobalGrid
using TinyKernels

using GLMakie

include("load_dem.jl")
include("signed_distances.jl")
include("level_sets.jl")
include("volume_fractions.jl")
include("bcs.jl")
include("stokes.jl")
include("data_io.jl")
include("hide_communication.jl")

@views av1(A) = 0.5.*(A[1:end-1].+A[2:end])
@views inn_x(A) = A[2:end-1,:,:]
@views inn_y(A) = A[:,2:end-1,:]
@views inn_z(A) = A[:,:,2:end-1]
@views inn(A)   = A[2:end-1,2:end-1,2:end-1]

@views function main(grid_dims)
    # path to DEM data
    greenland_path = "data/BedMachine/greenland.jld2"

    # region to simulate
    global_region = (xlims = (1100.0e3,1200.0e3), ylims = (1000.0e3,1100.0e3))

    # load DEM
    @info "loading DEM data from the file '$greenland_path'"
    (;x,y,bed,surface) = load_dem(greenland_path,global_region)
    @info "DEM resolution: $(size(bed,1)) × $(size(bed,2))"

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

    ρg  = (x=0.0,y=0.0,z=1.0)

    # local domain size and origin
    lx_l,ly_l,lz_l = (lx_g,ly_g,lz_g)./dims
    ox_l,oy_l,oz_l = (ox_g,oy_g,oz_g) .+ coords.*(lx_l,ly_l,lz_l)

    # numerics
    nx,ny,nz = grid_dims
    dx,dy,dz = lx_g/nx_g(), ly_g/ny_g(), lz_g/nz_g()
    bwidth   = (8,4,4)

    # preprocessing
    xv_l = LinRange(ox_l,ox_l+lx_l,nx+1)
    yv_l = LinRange(oy_l,oy_l+ly_l,ny+1)
    zv_l = LinRange(oz_l,oz_l+lz_l,nz+1)
    xc_l,yc_l,zc_l = av1.((xv_l,yv_l,zv_l))

    # PT params
    r          = 0.7
    lτ_re_mech = 0.2min(lx_g,ly_g,lz_g)/π
    vdτ        = min(dx,dy,dz)/sqrt(10.1)
    θ_dτ       = lτ_re_mech*(r+4/3)/vdτ
    nudτ       = vdτ*lτ_re_mech
    dτ_r       = 1.0/(θ_dτ+1.0)

    # fields allocation
    # level set
    Ψ = (
        fluid  = field_array(Float64,nx+1,ny+1,nz+1),
        liquid = field_array(Float64,nx+1,ny+1,nz+1),
    )
    wt = (
        fluid = (
            c  = field_array(Float64,nx  ,ny  ,nz  ),
            x  = field_array(Float64,nx+1,ny  ,nz  ),
            y  = field_array(Float64,nx  ,ny+1,nz  ),
            z  = field_array(Float64,nx  ,ny  ,nz+1),
            xy = field_array(Float64,nx-1,ny-1,nz-2),
            xz = field_array(Float64,nx-1,ny-2,nz-1),
            yz = field_array(Float64,nx-2,ny-1,nz-1),
        ),
        liquid = (
            c  = field_array(Float64,nx  ,ny  ,nz  ),
            x  = field_array(Float64,nx+1,ny  ,nz  ),
            y  = field_array(Float64,nx  ,ny+1,nz  ),
            z  = field_array(Float64,nx  ,ny  ,nz+1),
            xy = field_array(Float64,nx-1,ny-1,nz-2),
            xz = field_array(Float64,nx-1,ny-2,nz-1),
            yz = field_array(Float64,nx-2,ny-1,nz-1),
        )
    )
    # mechanics
    Pr = field_array(Float64,nx,ny,nz)
    τ  = (
        xx = field_array(Float64,nx  ,ny  ,nz  ),
        yy = field_array(Float64,nx  ,ny  ,nz  ),
        zz = field_array(Float64,nx  ,ny  ,nz  ),
        xy = field_array(Float64,nx-1,ny-1,nz-2),
        xz = field_array(Float64,nx-1,ny-2,nz-1),
        yz = field_array(Float64,nx-2,ny-1,nz-1),
    )
    V = (
        x = field_array(Float64,nx+1,ny,nz),
        y = field_array(Float64,nx,ny+1,nz),
        z = field_array(Float64,nx,ny,nz+1),
    )
    ηs = field_array(Float64,nx,ny,nz)
    # residuals
    Res = (
        Pr = field_array(Float64,nx,ny,nz),
        V = (
            x = field_array(Float64,nx-1,ny-2,nz-2),
            y = field_array(Float64,nx-2,ny-1,nz-2),
            z = field_array(Float64,nx-2,ny-2,nz-1),
        )
    )

    # figures
    fig = Figure(resolution=(1500,700),fontsize=32)
    axs = (
        dem = (
            bed     = Axis3(fig[1,1][1,1];title="bed"    ,aspect=:data),
            surface = Axis3(fig[1,2][1,1];title="surface",aspect=:data),
        ),
        Ψ = (
            fluid  = Axis3(fig[2,1][1,1];title="fluid" ,aspect=:data),
            liquid = Axis3(fig[2,2][1,1];title="liquid",aspect=:data),
        ),
        wt = (
            fluid  = Axis(fig[3,1][1,1];title="fluid" ),
            liquid = Axis(fig[3,2][1,1];title="liquid"),
        ),
        Pr = Axis(fig[4,1][1,1];title="Pr"),
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
        wt = (
            fluid  = heatmap!(axs.wt.fluid ,xc_l,zc_l,Array(wt.fluid.c[:,64,:] );colormap=:grays),
            liquid = heatmap!(axs.wt.liquid,xc_l,zc_l,Array(wt.liquid.c[:,64,:]);colormap=:grays),
        ),
        Pr = heatmap!(axs.Pr,xc_l,zc_l,Array(Pr[:,64,:]);colormap=:turbo),
    )
    Colorbar(fig[1,1][1,2],plts.dem.bed)
    Colorbar(fig[1,2][1,2],plts.dem.surface)
    Colorbar(fig[3,1][1,2],plts.wt.fluid)
    Colorbar(fig[3,2][1,2],plts.wt.liquid)
    Colorbar(fig[4,1][1,2],plts.Pr)

    # initialisation
    for comp in eachindex(V) fill!(V[comp],0.0) end
    for comp in eachindex(τ) fill!(τ[comp],0.0) end
    fill!(Pr,0.0)
    fill!(ηs,1.0)

    # compute level sets from DEM data
    dem_grid = (dem_data.x,dem_data.y)
    Ψ_grid   = (xv_l,yv_l,zv_l)
    
    (me==0) && @info "computing the level set for the ice surface"
    compute_level_set_from_dem!(Ψ.liquid,to_device(dem_data.surface),dem_grid,Ψ_grid)

    (me==0) && @info "computing the level set for the bedrock surface"
    compute_level_set_from_dem!(Ψ.fluid,to_device(dem_data.bed),dem_grid,Ψ_grid)
    TinyKernels.device_synchronize(get_device())
    @. Ψ.fluid *= -1.0
    TinyKernels.device_synchronize(get_device())

    @info "computing volume fractions from level sets"
    for phase in eachindex(Ψ)
        compute_volume_fractions_from_level_set!(wt[phase],Ψ[phase],dx,dy,dz)
    end
    
    # update plots
    plts.Ψ.fluid[4]   = Array(Ψ.fluid)
    plts.Ψ.liquid[4]  = Array(Ψ.liquid)
    plts.wt.fluid[3]  = Array(wt.fluid.c[:,256,:] )
    plts.wt.liquid[3] = Array(wt.liquid.c[:,256,:])
    display(fig)

    (me==0) && @info "iteration loop"
    for iter in 1:1000
        println("  iter: $iter")
        update_σ!(Pr,τ,V,ηs,wt,r,θ_dτ,dτ_r,dx,dy,dz)
        update_V!(V,Pr,τ,ηs,wt,nudτ,ρg,dx,dy,dz;bwidth)
        if iter % 50 == 0
            plts.Pr[3] = Array(Pr[:,256,:])
            yield()
        end
    end

    (me==0) && @info "saving results on disk"
    dim_g = (nx_g()-2, ny_g()-2, nz_g()-2)
    update_vis_fields!(Vmag,τII,V,τ)
    out_h5 = "results.h5"
    ndrange = CartesianIndices(( (coords[1]*(nx-2) + 1):(coords[1]+1)*(nx-2),
                                 (coords[2]*(ny-2) + 1):(coords[2]+1)*(ny-2),
                                 (coords[3]*(nz-2) + 1):(coords[3]+1)*(nz-2) ))
    fields = Dict("LS_ice"=>inn(Ψ.liquid),"LS_bed"=>inn(Ψ.fluid),"Vmag"=>Vmag,"TII"=>τII,"Pr"=>inn(Pr))
    (me==0) && @info "saving HDF5 file"
    write_h5(out_h5,fields,dim_g,ndrange,comm_cart,MPI.Info())

    if me==0
        @info "saving XDMF file..."
        write_xdmf("results.xdmf3",out_h5,fields,(xc_l[2],yc_l[2],zc_l[2]),(dx,dy,dz),dim_g)
    end

    finalize_global_grid(;finalize_MPI=false)
    MPI.Finalize()
    return
end

@tiny function _kernel_update_vis_fields!(Vmag, τII, V, τ)
    ix,iy,iz = @indices
    @inline isin(A) = checkbounds(Bool,A,ix,iy,iz)
    @inbounds if isin(Vmag)
        vxc = 0.5*(V.x[ix+1,iy+1,iz+1] + V.x[ix+2,iy+1,iz+1])
        vyc = 0.5*(V.y[ix+1,iy+1,iz+1] + V.y[ix+1,iy+2,iz+1])
        vzc = 0.5*(V.z[ix+1,iy+1,iz+1] + V.z[ix+1,iy+1,iz+2])
        Vmag[ix,iy,iz] = sqrt(vxc^2 + vyc^2 + vzc^2)
    end
    @inbounds if isin(τII)
        τxyc = 0.25*(τxy[ix,iy,iz]+τxy[ix+1,iy,iz]+τxy[ix,iy+1,iz]+τxy[ix+1,iy+1,iz])
        τxzc = 0.25*(τxz[ix,iy,iz]+τxz[ix+1,iy,iz]+τxz[ix,iy,iz+1]+τxz[ix+1,iy,iz+1])
        τyzc = 0.25*(τyz[ix,iy,iz]+τyz[ix,iy+1,iz]+τyz[ix,iy,iz+1]+τyz[ix,iy+1,iz+1])
        τII[ix,iy,iz] = sqrt(0.5*(τ.xx[ix+1,iy+1,iz+1]^2 + τ.yy[ix+1,iy+1,iz+1]^2 + τ.zz[ix+1,iy+1,iz+1]^2) + τxyc^2 + τxzc^2 + τyzc^2)
    end
    return
end

const _update_vis_fields! = Kernel(_kernel_update_vis_fields!,get_device())

function update_vis_fields!(Vmag, τII, V, τ)
    wait(_update_vis_fields!(Vmag, τII, V, τ; ndrange=axes(Vmag)))
    return
end

main((512,512,32))