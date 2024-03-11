using Chmy.Architectures
using Chmy.Grids
using Chmy.Fields

using FastIce.LevelSets
using FastIce.Geometries

using KernelAbstractions

# using CUDA
# using AMDGPU

# Select backend
backend = CPU()
# backend = CUDABackend()
# backend = ROCBackend()

using Chmy.Distributed
using MPI
MPI.Init()

# function main(backend=CPU(); nxyz_l=(126, 126))
    backend=CPU()
    nxyz_l=(126, 126)
    arch  = Arch(backend, MPI.COMM_WORLD, (0, 0, 0))
    arch2 = Arch(backend, MPI.COMM_WORLD, (0, 0))
    topo = topology(arch)
    me   = global_rank(topo)
    # geometry
    zmin, zmax = 0.0, 1.0
    lx, ly, lz = 5.0, 5.0, zmax - zmin
    nx, ny = nxyz_l[1:2]
    nz = (length(nxyz_l) < 3) ? ceil(Int, lz / lx * nxyz_l[1]) : nxyz_l[3]
    dims_l     = (nx, ny, nz)
    dims_g     = dims_l .* dims(topo)
    grid       = UniformGrid(arch; origin=(-lx/2, -ly/2, -lz/2), extent=(lx, ly, lz), dims=dims_g)
    grid_dem   = UniformGrid(arch2; origin=(-lx/2, -ly/2), extent=(lx, ly), dims=dims_g[1:2])
    # synthetic topo
    amp  = 0.1
    ω    = 10π
    tanβ = tan(-π/6)
    el   = 0.35
    gl   = 0.9
    # bed and surf functions
    # type = :turtle
    generate_surf(x, y, gl, lx, ly) = gl * (1.0 - ((1.7 * x / lx + 0.22)^2 + (0.5 * y / ly)^2))
    generate_bed(x, y, amp, ω, tanβ, el, lx, ly, lz) = lz * (amp * sin(ω * x / lx) * sin(ω * y /ly) + tanβ * x / lx + el + (1.5 * y / ly)^2)
    # init fields
    dem_surf = Field(backend, grid_dem, Vertex())
    dem_bed  = Field(backend, grid_dem, Vertex())
    Ψ = (na=Field(backend, grid, Vertex()),
         ns=Field(backend, grid, Vertex()))
    wt = (na=volfrac_field(backend, grid),
          ns=volfrac_field(backend, grid))
    # generate 2D synthetic dem
    set!(dem_surf, grid_dem, generate_surf; parameters=(gl=gl, lx=lx, ly=ly))
    set!(dem_bed, grid_dem, generate_bed; parameters=(amp=amp, ω=ω, tanβ=tanβ, el=el, lx=lx, ly=ly, lz=lz))
    # comput level set
    compute_level_set_from_dem!(arch, Ψ.na, dem_surf, grid_dem, grid)
    compute_level_set_from_dem!(arch, Ψ.ns, dem_bed, grid_dem, grid)
    # invert level set to set what's below the DEM surface as inside
    @. Ψ.ns *= -1.0
    # volume fractions
    for phase in eachindex(Ψ)
        compute_volfrac_from_level_set!(arch, wt[phase], Ψ[phase], grid)
    end
    # gather
    dem_bed_v  = (me==0) ? KernelAbstractions.zeros(CPU(), Float64, size(interior(dem_bed))  .* dims(topo)[1:2]) : nothing
    dem_surf_v = (me==0) ? KernelAbstractions.zeros(CPU(), Float64, size(interior(dem_surf)) .* dims(topo)[1:2]) : nothing
    wt_na_c = (me==0) ? KernelAbstractions.zeros(CPU(), Float64, size(interior(wt.na.c)) .* dims(topo)) : nothing
    wt_ns_c = (me==0) ? KernelAbstractions.zeros(CPU(), Float64, size(interior(wt.ns.c)) .* dims(topo)) : nothing
    gather!(arch2, dem_bed_v, dem_bed)
    gather!(arch2, dem_surf_v, dem_surf)
    gather!(arch, wt_na_c, wt.na.c)
    gather!(arch, wt_ns_c, wt.ns.c)
    # visualise
    dem_surf_v[dem_surf_v.<dem_bed_v] .= NaN
    slx = ceil(Int, size(wt_na_c, 1) / 2) # for visu
    sly = ceil(Int, size(wt_na_c, 2) / 2) # for visu
    x_g = LinRange(-lx / 2, lx / 2, size(dem_bed_v, 1))
    y_g = LinRange(-ly / 2, ly / 2, size(dem_bed_v, 2))
    # z_g = LinRange(-lz / 2, lz / 2, size(wt_na_c, 3) + 1)

    fig = Figure(; size=(1000, 800), fontsize=22)
    axs = (ax1 = Axis3(fig[1, 1][1, 1]; aspect=(2, 2, 1), azimuth=-π / 8, elevation=π / 5),
           ax2 = Axis(fig[1, 2]; aspect=DataAspect()),
           ax3 = Axis(fig[2, 1]; aspect=DataAspect()),
           ax4 = Axis(fig[2, 2]; aspect=DataAspect()),
           ax5 = Axis(fig[3, 1]; aspect=DataAspect()),
           ax6 = Axis(fig[3, 2]; aspect=DataAspect()))
    plt = (p1 = surface!(axs.ax1, x_g, y_g, dem_bed_v; colormap=:turbo),
           p1_= surface!(axs.ax1, x_g, y_g, dem_surf_v; colormap=:turbo),
           p2 = plot!(axs.ax2, x_g, dem_bed_v[:, sly]),
           p2_= plot!(axs.ax2, x_g, dem_surf_v[:, sly]),
           p3 = heatmap!(axs.ax3, wt_na_c[:, sly, :]; colormap=:turbo),
           p4 = heatmap!(axs.ax4, wt_ns_c[:, sly, :]; colormap=:turbo),
           p5 = heatmap!(axs.ax5, wt_na_c[slx, :, :]; colormap=:turbo),
           p6 = heatmap!(axs.ax6, wt_ns_c[slx, :, :]; colormap=:turbo))
    Colorbar(fig[1, 1][1, 2], plt.p1)
    Colorbar(fig[2, 2][1, 2], plt.p4)
    display(fig)

#     return
# end
