using Chmy.Architectures
using Chmy.Grids
using Chmy.Fields
using Chmy.GridOperators
using Chmy.KernelLaunch

using FastIce.Writers
using FastIce.LevelSets
using FastIce.Geometries
using JLD2

using KernelAbstractions
using CairoMakie
using CUDA
# using AMDGPU
# AMDGPU.allowscalar(false)

# Select backend
# backend = CPU()
backend = CUDABackend()
# backend = ROCBackend()

do_h5_save = true

using Chmy.Distributed
using MPI
MPI.Init()

conv(nx, tx) = tx * ((nx + tx ÷ 2 -1 ) ÷ tx)

function make_synthetic(arch::Architecture, nx, ny, lx, ly, lz, amp, ω, tanβ, el, gl)
    arch_2d = SingleDeviceArchitecture(arch)
    grid_2d = UniformGrid(arch_2d; origin=(-lx/2, -ly/2), extent=(lx, ly), dims=(nx, ny))

    # type = :turtle
    generate_ice(x, y, gl, lx, ly) = gl * (1.0 - ((1.7 * x / lx + 0.22)^2 + (0.5 * y / ly)^2))
    generate_bed(x, y, amp, ω, tanβ, el, lx, ly, lz) = lz * (amp * sin(ω * x / lx) * sin(ω * y /ly) + tanβ * x / lx + el + (1.5 * y / ly)^2)

    ice = FunctionField(generate_ice, grid_2d, Vertex(); parameters=(; gl, lx, ly))
    bed  = FunctionField(generate_bed, grid_2d, Vertex(); parameters=(; amp, ω, tanβ, el, lx, ly, lz))

    return (; arch, bed, ice, grid_2d, lx, ly, lz)
end

function main_synthetic(backend=CPU())
    # set-up distributed
    arch = Arch(backend, MPI.COMM_WORLD, (0, 0, 0))

    # synthetic topo
    lx, ly, lz = 5.0, 5.0, 1.0
    amp  = 0.1
    ω    = 10π
    tanβ = tan(-π/6)
    el   = 0.35
    gl   = 0.9

    nx, ny = 126, 126
    nz     = max(conv(ceil(Int, lz / lx * nx), 30), 30)
    resol  = (nx, ny, nz)

    synthetic_elevation = make_synthetic(arch, nx, ny, lx, ly, lz, amp, ω, tanβ, el, gl)

    run_simulation(synthetic_elevation..., resol...)
    return
end

function extract_dem(arch::Architecture, data_path::String)
    data  = load(data_path)
    dtype = first(keys(data))

    z_ice      = data[dtype].z_surf
    z_bed      = data[dtype].z_bed
    dm         = data[dtype].domain
    dm         = dilate(dm, 0.05)
    lx, ly, lz = extents(dm)
    nx, ny     = size(z_ice) .- 1

    z_ice .-= dm.zmin # put Z-origin at 0
    z_bed .-= dm.zmin # put Z-origin at 0

    arch_2d = SingleDeviceArchitecture(arch)
    grid_2d = UniformGrid(arch_2d; origin=(-lx/2, -ly/2), extent=(lx, ly), dims=(nx, ny))

    ice = Field(arch_2d, grid_2d, Vertex())
    bed  = Field(arch_2d, grid_2d, Vertex())
    set!(ice, z_ice)
    set!(bed, z_bed)

    return (; arch, bed, ice, grid_2d, lx, ly, lz)
end

function main_vavilov(backend=CPU())
    # load dem data
    data_path = "./vavilov_dem2.jld2"

    # set-up distributed
    arch = Arch(backend, MPI.COMM_WORLD, (0, 0, 0))

    data_elevation = extract_dem(arch, data_path)

    nx, ny = 126, 126
    nz     = max(conv(ceil(Int, data_elevation.lz / data_elevation.lx * nx), 30), 60)
    resol  = (nx, ny, nz)

    run_simulation(data_elevation..., resol...)
    return
end

@views function run_simulation(arch::Architecture, bed, ice, grid_2d, lx, ly, lz, nx, ny, nz)
    # distributed arch we get from the outside
    topo = topology(arch)
    me   = global_rank(topo)
    # geometry
    (me == 0) && @info "size = ($nx, $ny, $nz), extent = ($lx, $ly, $lz)"
    dims_l = (nx, ny, nz)
    dims_g = dims_l .* dims(topo)
    grid   = UniformGrid(arch; origin=(-lx/2, -ly/2, 0.0), extent=(lx, ly, lz), dims=dims_g)
    launch = Launcher(arch, grid; outer_width=(16, 8, 4))
    # init fields
    Ψ = (na=Field(arch, grid, Vertex()),
         ns=Field(arch, grid, Vertex()))
    wt = (na=FieldMask(arch, grid),
          ns=FieldMask(arch, grid))
    # comput level set
    compute_levelset_from_dem!(arch, launch, Ψ.na, ice, grid_2d, grid)
    compute_levelset_from_dem!(arch, launch, Ψ.ns, bed, grid_2d, grid)
    invert_levelset!(arch, launch, Ψ.ns, grid) # invert level set to set what's below the DEM surface as inside
    # volume fractions
    for phase in eachindex(Ψ)
        ω_from_ψ!(arch, launch, wt[phase], Ψ[phase], grid)
    end

    # compute physics or else

    # postprocessing
    wt_na_c = (me == 0) ? KernelAbstractions.zeros(CPU(), Float64, size(interior(wt.na.ccc)) .* dims(topo)) : nothing
    wt_ns_c = (me == 0) ? KernelAbstractions.zeros(CPU(), Float64, size(interior(wt.ns.ccc)) .* dims(topo)) : nothing
    gather!(arch, wt_na_c, wt.na.ccc)
    gather!(arch, wt_ns_c, wt.ns.ccc)
    # visualise
    if me == 0
        dem_bed = Array(interior(bed))
        dem_ice = Array(interior(ice))
        dem_ice[dem_ice .< dem_bed] .= NaN
        slx = ceil(Int, size(wt_na_c, 1) / 2) # for visu
        sly = ceil(Int, size(wt_na_c, 2) / 2) # for visu
        x_g = LinRange(-lx / 2, lx / 2, size(dem_bed, 1))
        y_g = LinRange(-ly / 2, ly / 2, size(dem_bed, 2))

        fig = Figure(; size=(1000, 800), fontsize=22)
        axs = (ax1 = Axis3(fig[1, 1][1, 1]; aspect=(2, 2, 1), azimuth=-π / 8, elevation=π / 5),
               ax2 = Axis(fig[1, 2]; aspect=DataAspect()),
               ax3 = Axis(fig[2, 1]; aspect=DataAspect()),
               ax4 = Axis(fig[2, 2][1, 1]; aspect=DataAspect()),
               ax5 = Axis(fig[3, 1]; aspect=DataAspect()),
               ax6 = Axis(fig[3, 2]; aspect=DataAspect()))
        plt = (p1  = surface!(axs.ax1, x_g, y_g, dem_bed; colormap=:turbo),
               p1_ = surface!(axs.ax1, x_g, y_g, dem_ice; colormap=:turbo),
               p2  = plot!(axs.ax2, x_g, dem_bed[:, sly] |> Array),
               p2_ = plot!(axs.ax2, x_g, dem_ice[:, sly] |> Array),
               p3  = heatmap!(axs.ax3, wt_na_c[:, sly, :]; colormap=:turbo),
               p4  = heatmap!(axs.ax4, wt_ns_c[:, sly, :]; colormap=:turbo),
               p5  = heatmap!(axs.ax5, wt_na_c[slx, :, :]; colormap=:turbo),
               p6  = heatmap!(axs.ax6, wt_ns_c[slx, :, :]; colormap=:turbo))
        Colorbar(fig[1, 1][1, 2], plt.p1)
        Colorbar(fig[2, 2][1, 2], plt.p4)
        # display(fig)
        save("levset_$(dims_g[1])_v.png", fig)
    end

    if do_h5_save
        h5names = String[]
        fields = Dict("wt_na" => wt.na.ccc, "wt_ns" => wt.ns.ccc)
        outdir = "out_visu_mpi_v"
        (me == 0) && mkpath(outdir)

        out_h5 = "results.h5"
        (me == 0) && @info "saving HDF5 file"
        write_h5(arch, grid, joinpath(outdir, out_h5), fields)
        push!(h5names, out_h5)

        (me == 0) && @info "saving XDMF file"
        (me == 0) && write_xdmf(arch, grid, joinpath(outdir, "results.xdmf3"), fields, h5names)
    end

    return
end

# main_synthetic(backend)
main_vavilov(backend)

# MPI.Finalize()
