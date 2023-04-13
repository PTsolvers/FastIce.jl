using FastIce
using Logging
using MPI
using ImplicitGlobalGrid
using TinyKernels
using HDF5
using LightXML
using CairoMakie

include("load_dem.jl")
include("signed_distances.jl")
include("level_sets.jl")
include("volume_fractions.jl")
include("bcs.jl")
include("stokes.jl")
include("data_io.jl")
include("hide_communication.jl")

@views av1(A) = 0.5 .* (A[1:end-1] .+ A[2:end])
@views inn_x(A) = A[2:end-1, :, :]
@views inn_y(A) = A[:, 2:end-1, :]
@views inn_z(A) = A[:, :, 2:end-1]
@views inn(A) = A[2:end-1, 2:end-1, 2:end-1]

@views function main(grid_dims,grid)
    # unpack values
    me, dims, nprocs, coords, comm_cart = grid

    # init logger
    global_logger(FastIce.Logging.MPILogger(0, comm_cart, global_logger()))

    # path to DEM data
    greenland_path = "data/BedMachine/greenland.jld2"

    # region to simulate
    global_region = (xlims=(1100.0e3, 1200.0e3), ylims=(1000.0e3, 1100.0e3))

    # load DEM
    @info "loading DEM data from the file '$greenland_path'"
    (; x, y, bed, surface) = load_dem(greenland_path, global_region)
    @info "DEM resolution: $(size(bed,1)) × $(size(bed,2))"

    @info "plot DEMs"
    if me == 0
        fig = Figure(resolution=(2000,700),fontsize=32)
        ax  = (
            bed = Axis(fig[1,1][1,1];aspect=DataAspect(),title="bedrock",xlabel="x",ylabel="y"),
            ice = Axis(fig[1,2][1,1];aspect=DataAspect(),title="ice"    ,xlabel="x",ylabel="y"),
        )
        plt = (
            bed = heatmap!(ax.bed,x,y,bed    ;colormap=:terrain),
            ice = heatmap!(ax.ice,x,y,surface;colormap=:terrain),
        )
        Colorbar(fig[1,1][1,2],plt.bed)
        Colorbar(fig[1,2][1,2],plt.ice)
        save("region.png",fig)
    end

    # compute origin and size of the domain (required for scaling and computing the grid size)
    ox, oy, oz = x[1], y[1], minimum(bed)
    lx = x[end] - ox
    ly = y[end] - oy
    lz = maximum(surface) - oz

    # shift and scale the domain before computation (center of the domain is (0,0) in x-y plane)
    δx, δy = ox + 0.5lx, oy + 0.5ly # required to avoid conversion to Vector  
    x = @. (x - δx) / lz
    y = @. (y - δy) / lz
    @. bed = (bed - oz) / lz
    @. surface = (surface - oz) / lz

    @. surface -= 0.05

    # run simulation
    dem_data = (; x, y, bed, surface)
    @info "running the simulation"
    run_simulation(dem_data, grid_dims, me, dims, coords, comm_cart)

    return
end

@views function run_simulation(dem_data, grid_dims, me, dims, coords, comm_cart)
    # physics
    # global domain origin and size
    ox_g, oy_g, oz_g = dem_data.x[1], dem_data.y[1], 0.0
    lx_g = dem_data.x[end] - ox_g
    ly_g = dem_data.y[end] - oy_g
    lz_g = 1.0

    ρg = (x=0.0, y=0.0, z=1.0)

    # local domain size and origin
    lx_l, ly_l, lz_l = (lx_g, ly_g, lz_g) ./ dims
    ox_l, oy_l, oz_l = (ox_g, oy_g, oz_g) .+ coords .* (lx_l, ly_l, lz_l)

    # numerics
    nx, ny, nz       = grid_dims
    nx_l, ny_l, nz_l = grid_dims .+ 2 # include ghost nodes
    nx_g, ny_g, nz_g = grid_dims.*dims
    bwidth = (8, 4, 4)

    # preprocessing
    dx, dy, dz = lx_g / nx_g, ly_g / ny_g, lz_g / nz_g
    @info "grid spacing: dx = $dx, dy = $dy, dz = $dz"

    # take into account ghost nodes to simplify model setup
    xv_l = LinRange(ox_l - dx, ox_l + lx_l + dx, nx_l + 1)
    yv_l = LinRange(oy_l - dy, oy_l + ly_l + dy, ny_l + 1)
    zv_l = LinRange(oz_l - dz, oz_l + lz_l + dz, nz_l + 1)
    xc_l, yc_l, zc_l = av1.((xv_l, yv_l, zv_l))

    # PT params
    r = 0.7
    lτ_re_mech = 0.5min(lx_g, ly_g, lz_g) / π
    vdτ = min(dx, dy, dz) / sqrt(5.1)
    θ_dτ = lτ_re_mech * (r + 4 / 3) / vdτ
    nudτ = vdτ * lτ_re_mech
    dτ_r = 1.0 / (θ_dτ + 1.0)

    # fields allocation
    # level set
    Ψ = (
        not_solid=scalar_field(Float64, nx_l + 1, ny_l + 1, nz_l + 1),
        not_air  =scalar_field(Float64, nx_l + 1, ny_l + 1, nz_l + 1),
    )
    wt = (
        not_solid=volfrac_field(Float64, nx_l, ny_l, nz_l),
        not_air  =volfrac_field(Float64, nx_l, ny_l, nz_l),
    )
    # mechanics (stress fields include ghost nodes due to redundant computations on distributed staggered grid)
    Pr = scalar_field(Float64, nx_l, ny_l, nz_l)
    τ  = tensor_field(Float64, nx_l, ny_l, nz_l)
    V  = vector_field(Float64, nx_l, ny_l, nz_l)
    ηs = scalar_field(Float64, nx_l, ny_l, nz_l)
    # residuals
    Res = (
        Pr=scalar_field(Float64, nx, ny, nz),
        V =vector_field(Float64, nx, ny, nz)
    )
    # visualisation
    Vmag = scalar_field(Float64, nx, ny, nz)
    τII  = scalar_field(Float64, nx, ny, nz)
    Ψav = (
        not_air=scalar_field(Float64,nx,ny,nz),
        not_solid=scalar_field(Float64,nx,ny,nz),
    )

    # initialisation
    for comp in eachindex(V) fill!(V[comp], 0.0) end
    for comp in eachindex(τ) fill!(τ[comp], 0.0) end
    fill!(Pr, 0.0)
    fill!(ηs, 1.0)

    # compute level sets from DEM data
    dem_grid = (dem_data.x, dem_data.y)
    Ψ_grid = (xv_l, yv_l, zv_l)

    @info "computing the level set for the ice surface"
    compute_level_set_from_dem!(Ψ.not_air, to_device(dem_data.surface), dem_grid, Ψ_grid)

    @info "computing the level set for the bedrock surface"
    compute_level_set_from_dem!(Ψ.not_solid, to_device(dem_data.bed), dem_grid, Ψ_grid)
    TinyKernels.device_synchronize(get_device())
    # invert level set to set what's below the DEM surface as inside
    @. Ψ.not_solid *= -1.0
    TinyKernels.device_synchronize(get_device())

    @info "computing volume fractions from level sets"
    for phase in eachindex(Ψ)
        compute_volume_fractions_from_level_set!(wt[phase], Ψ[phase], dx, dy, dz)
    end

    @info "iteration loop"
    for iter in 1:500
        @info "  iter: $iter"
        update_σ!(Pr, τ, V, ηs, wt, r, θ_dτ, dτ_r, dx, dy, dz)
        update_V!(V, Pr, τ, ηs, wt, nudτ, ρg, dx, dy, dz; bwidth)
    end

    @info "saving results on disk"
    dim_g = (nx_g, ny_g, nz_g)
    update_vis_fields!(Vmag, τII, Ψav, V, τ, Ψ)
    out_h5 = "results.h5"
    ndrange = CartesianIndices(((coords[1]*nx+1):(coords[1]+1)*nx,
                                (coords[2]*ny+1):(coords[2]+1)*ny,
                                (coords[3]*nz+1):(coords[3]+1)*nz))
    fields = Dict("LS_ice" => Ψav.not_air, "LS_bed" => Ψav.not_solid, "Vmag" => Vmag, "TII" => τII, "Pr" => inn(Pr))
    @info "saving HDF5 file"
    write_h5(out_h5, fields, dim_g, ndrange)

    @info "saving XDMF file..."
    (me == 0) && write_xdmf("results.xdmf3", out_h5, fields, (xc_l[2], yc_l[2], zc_l[2]), (dx, dy, dz), dim_g)

    return
end

@tiny function _kernel_update_vis_fields!(Vmag, τII, Ψav, V, τ, Ψ)
    ix, iy, iz = @indices
    @inline isin(A) = checkbounds(Bool, A, ix, iy, iz)
    @inbounds if isin(Ψ.not_air)
        pav = 0.0
        for idz = 1:2, idy = 1:2, idx = 1:2
            pav += Ψ.not_air[ix+idx, iy+idy, iz+idz]
        end
        Ψav.not_air[ix, iy, iz] = pav / 8
    end
    @inbounds if isin(Ψ.not_solid)
        pav = 0.0
        for idz = 1:2, idy = 1:2, idx = 1:2
            pav += Ψ.not_solid[ix+idx, iy+idy, iz+idz]
        end
        Ψav.not_solid[ix, iy, iz] = pav / 8
    end
    @inbounds if isin(Vmag)
        vxc = 0.5 * (V.x[ix+1, iy+1, iz+1] + V.x[ix+2, iy+1, iz+1])
        vyc = 0.5 * (V.y[ix+1, iy+1, iz+1] + V.y[ix+1, iy+2, iz+1])
        vzc = 0.5 * (V.z[ix+1, iy+1, iz+1] + V.z[ix+1, iy+1, iz+2])
        Vmag[ix, iy, iz] = sqrt(vxc^2 + vyc^2 + vzc^2)
    end
    @inbounds if isin(τII)
        τxyc = 0.25 * (τ.xy[ix, iy, iz] + τ.xy[ix+1, iy, iz] + τ.xy[ix, iy+1, iz] + τ.xy[ix+1, iy+1, iz])
        τxzc = 0.25 * (τ.xz[ix, iy, iz] + τ.xz[ix+1, iy, iz] + τ.xz[ix, iy, iz+1] + τ.xz[ix+1, iy, iz+1])
        τyzc = 0.25 * (τ.yz[ix, iy, iz] + τ.yz[ix, iy+1, iz] + τ.yz[ix, iy, iz+1] + τ.yz[ix, iy+1, iz+1])
        τII[ix, iy, iz] = sqrt(0.5 * (τ.xx[ix+1, iy+1, iz+1]^2 + τ.yy[ix+1, iy+1, iz+1]^2 + τ.zz[ix+1, iy+1, iz+1]^2) + τxyc^2 + τxzc^2 + τyzc^2)
    end
    return
end

const _update_vis_fields! = _kernel_update_vis_fields!(get_device())

function update_vis_fields!(Vmag, τII, Ψav, V, τ, Ψ)
    wait(_update_vis_fields!(Vmag, τII, Ψav, V, τ, Ψ; ndrange=axes(Vmag)))
    return
end

grid_dims = (1000, 1000, 50)

# init MPI and IGG
MPI.Init()
me, dims, nprocs, coords, comm_cart = init_global_grid(grid_dims...; init_MPI=false)
dims   = Tuple(dims)
coords = Tuple(coords)
grid   = (me,dims,nprocs,coords,comm_cart)

main(grid_dims,grid)

# finalize_global_grid(; finalize_MPI=false)
# MPI.Finalize()