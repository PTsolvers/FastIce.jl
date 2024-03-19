using Chmy.Architectures
using Chmy.Grids
using Chmy.Fields
using Chmy.KernelLaunch

using FastIce.LevelSets

using KernelAbstractions
using CairoMakie
# using CUDA
# using AMDGPU
# AMDGPU.allowscalar(false)

# Select backend
backend = CPU()
# backend = CUDABackend()
# backend = ROCBackend()

using Chmy.Distributed
using MPI
MPI.Init()

@views function make_dem(backend=CPU(); nx, ny, lx, ly, zmin, zmax, amp, ω, tanβ, el, gl)
    arch_2d = Arch(backend)
    grid_2d = UniformGrid(arch_2d; origin=(-lx/2, -ly/2), extent=(lx, ly), dims=(nx, ny))

    lz = zmax - zmin

    # type = :turtle
    generate_surf(x, y, gl, lx, ly) = gl * (1.0 - ((1.7 * x / lx + 0.22)^2 + (0.5 * y / ly)^2))
    generate_bed(x, y, amp, ω, tanβ, el, lx, ly, lz) = lz * (amp * sin(ω * x / lx) * sin(ω * y /ly) + tanβ * x / lx + el + (1.5 * y / ly)^2)

    surf = FunctionField(generate_surf, grid_2d, Vertex(); parameters=(; gl, lx, ly))
    bed  = FunctionField(generate_bed, grid_2d, Vertex(); parameters=(; amp, ω, tanβ, el, lx, ly, lz))

    return (; bed, surf, grid_2d, lz)
end

@views function main(backend=CPU())
    lx, ly = 5.0, 5.0
    zmin, zmax = 0.0, 1.0
    nx = ny = 126
    # synthetic topo
    amp  = 0.1
    ω    = 10π
    tanβ = tan(-π/6)
    el   = 0.35
    gl   = 0.9

    dem_data = make_dem(backend; nx, ny, lx, ly, zmin, zmax, amp, ω, tanβ, el, gl)

    run_simulation(backend; nxyz_l=(nx, ny), dem_data)
    return
end

@views function run_simulation(backend=CPU(); nxyz_l=(126, 126), dem_data)
    arch    = Arch(backend, MPI.COMM_WORLD, (0, 0, 0))
    topo    = topology(arch)
    me      = global_rank(topo)
    # geometry
    lx, ly, lz = extent(dem_data.grid_2d, Vertex())..., dem_data.lz
    nx, ny     = nxyz_l[1:2]
    nz         = (length(nxyz_l) < 3) ? ceil(Int, lz / lx * nxyz_l[1]) : nxyz_l[3]
    @info "nz = $nz (lz = $lz)"
    dims_l     = (nx, ny, nz)
    dims_g     = dims_l .* dims(topo)
    grid       = UniformGrid(arch; origin=(-lx/2, -ly/2, 0.0), extent=(lx, ly, lz), dims=dims_g)
    launch     = Launcher(arch, grid; outer_width=(16, 8, 4))
    # init fields
    Ψ = (na=Field(backend, grid, Vertex()),
         ns=Field(backend, grid, Vertex()))
    wt = (na=volfrac_field(backend, grid),
          ns=volfrac_field(backend, grid))
    # comput level set
    compute_levelset_from_dem!(arch, launch, Ψ.na, dem_data.surf, dem_data.grid_2d, grid)
    compute_levelset_from_dem!(arch, launch, Ψ.ns, dem_data.bed, dem_data.grid_2d, grid)
    invert_levelset!(arch, launch, Ψ.ns, grid) # invert level set to set what's below the DEM surface as inside
    # volume fractions
    for phase in eachindex(Ψ)
        compute_volfrac_from_levelset!(arch, launch, wt[phase], Ψ[phase], grid)
    end

    # compute physics or else

    # postprocessing
    wt_na_c = (me == 0) ? KernelAbstractions.zeros(CPU(), Float64, size(interior(wt.na.c)) .* dims(topo)) : nothing
    wt_ns_c = (me == 0) ? KernelAbstractions.zeros(CPU(), Float64, size(interior(wt.ns.c)) .* dims(topo)) : nothing
    gather!(arch, wt_na_c, wt.na.c)
    gather!(arch, wt_ns_c, wt.ns.c)
    # visualise
    if me == 0
        dem_bed  = Array(interior(dem_data.bed))
        dem_surf = Array(interior(dem_data.surf))
        dem_surf[dem_surf .< dem_bed] .= NaN
        slx = ceil(Int, size(wt_na_c, 1) / 2) # for visu
        sly = ceil(Int, size(wt_na_c, 2) / 2) # for visu
        x_g = LinRange(-lx / 2, lx / 2, size(dem_bed, 1))
        y_g = LinRange(-ly / 2, ly / 2, size(dem_bed, 2))

        fig = Figure(; size=(1000, 800), fontsize=22)
        axs = (ax1 = Axis3(fig[1, 1][1, 1]; aspect=(2, 2, 1), azimuth=-π / 8, elevation=π / 5),
               ax2 = Axis(fig[1, 2]; aspect=DataAspect()),
               ax3 = Axis(fig[2, 1]; aspect=DataAspect()),
               ax4 = Axis(fig[2, 2]; aspect=DataAspect()),
               ax5 = Axis(fig[3, 1]; aspect=DataAspect()),
               ax6 = Axis(fig[3, 2]; aspect=DataAspect()))
        plt = (p1  = surface!(axs.ax1, x_g, y_g, dem_bed; colormap=:turbo),
               p1_ = surface!(axs.ax1, x_g, y_g, dem_surf; colormap=:turbo),
               p2  = plot!(axs.ax2, x_g, dem_bed[:, sly] |> Array),
               p2_ = plot!(axs.ax2, x_g, dem_surf[:, sly] |> Array),
               p3  = heatmap!(axs.ax3, wt_na_c[:, sly, :]; colormap=:turbo),
               p4  = heatmap!(axs.ax4, wt_ns_c[:, sly, :]; colormap=:turbo),
               p5  = heatmap!(axs.ax5, wt_na_c[slx, :, :]; colormap=:turbo),
               p6  = heatmap!(axs.ax6, wt_ns_c[slx, :, :]; colormap=:turbo))
        Colorbar(fig[1, 1][1, 2], plt.p1)
        Colorbar(fig[2, 2][1, 2], plt.p4)
        display(fig)
        # save("levset_$nx.png", fig)
    end
    return
end

main(backend)

MPI.Finalize()
