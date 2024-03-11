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

arch = Arch(backend)

function load_data(data_path)
    dat = load(data_path)
    return first(keys(dat)), dat
end


"""
    extract_dem(arch::Architecture, data_path::String, z_resol=nothing)

Load digital elevation model of surface and bedrock from (JLD2) file and initialise the grid.
"""
function extract_dem(arch::Architecture, data_path::String, z_resol=nothing)
    dtype, data = load_data(data_path)
    z_surf = data[dtype].z_surf
    z_bed  = data[dtype].z_bed
    dm     = data[dtype].domain
    lx, ly, lz = dm.xmax - dm.xmin, dm.ymax - dm.ymin, dm.zmax - dm.zmin
    nx, ny = size(z_surf) .- 1
    nz = isnothing(z_resol) ? ceil(Int, lz / lx * nx) : z_resol
    Ψ_grid   = UniformGrid(arch; origin=(dm.xmin, dm.ymin, dm.zmin), extent=(lx, ly, lz), dims=(nx, ny, nz))
    dem_grid = UniformGrid(arch; origin=(dm.xmin, dm.ymin), extent=(lx, ly), dims=(nx, ny))
    dem_surf = Field(arch.backend, dem_grid, Vertex())
    dem_bed  = Field(arch.backend, dem_grid, Vertex())
    set!(dem_surf, z_surf)
    set!(dem_bed, z_bed)
    return dem_surf, dem_bed, dem_grid, Ψ_grid
end

function extract_dem(arch::Architecture, data::SyntheticElevation, z_resol=nothing)
    z_surf = data.z_surf
    z_bed  = data.z_bed
    dm     = data.domain
    lx, ly, lz = extents(dm)
    nx, ny = size(z_surf) .- 1
    nz = isnothing(z_resol) ? ceil(Int, lz / lx * nx) : z_resol
    Ψ_grid   = UniformGrid(arch; origin=(dm.xmin, dm.ymin, dm.zmin), extent=(lx, ly, lz), dims=(nx, ny, nz))
    dem_grid = UniformGrid(arch; origin=(dm.xmin, dm.ymin), extent=(lx, ly), dims=(nx, ny))
    dem_surf = Field(arch.backend, dem_grid, Vertex())
    dem_bed  = Field(arch.backend, dem_grid, Vertex())
    set!(dem_surf, z_surf)
    set!(dem_bed, z_bed)
    dem = (surf=dem_surf, bed=dem_bed)
    return dem, dem_grid, Ψ_grid
end

nx, ny     = 128, 128
lx, ly     = 5.0, 5.0
zmin, zmax = 0.0, 1.0
type = :turtle

synth_dem = make_synthetic(nx, ny, lx, ly, zmin, zmax, type)
dem, dem_grid, Ψ_grid = extract_dem(arch, synth_dem)

Ψ = (
    na=Field(backend, Ψ_grid, Vertex()),
    ns=Field(backend, Ψ_grid, Vertex()),
)

compute_level_set_from_dem!(arch, Ψ.na, dem.surf, dem_grid, Ψ_grid)
compute_level_set_from_dem!(arch, Ψ.ns, dem.bed, dem_grid, Ψ_grid)
# invert level set to set what's below the DEM surface as inside
@. Ψ.ns *= -1.0

wt = (
    na=volfrac_field(backend, Ψ_grid),
    ns=volfrac_field(backend, Ψ_grid),
)

for phase in eachindex(Ψ)
    compute_volfrac_from_level_set!(arch, wt[phase], Ψ[phase], Ψ_grid)
end


@views function visme(synth_dem, Ψ, wt)
    nx, ny, nz = size(Ψ.ns)
    x = LinRange(synth_dem.domain.xmin, synth_dem.domain.xmax, nx)
    y = LinRange(synth_dem.domain.ymin, synth_dem.domain.ymax, ny)
    z = LinRange(synth_dem.domain.zmin, synth_dem.domain.zmax, nz)
    bed  = synth_dem.z_bed
    surf = synth_dem.z_surf

    surf2 = copy(surf)
    surf2[surf.<bed] .= NaN

    slx = ceil(Int, length(x) / 2) + 1
    sly = ceil(Int, length(y) / 2)

    fig = Figure(; size=(1000, 800), fontsize=22)
    ax1 = Axis3(fig[1, 1]; aspect=(2, 2, 1), azimuth=-π / 8, elevation=π / 5)
    ax2 = Axis(fig[1, 2]; aspect=DataAspect())
    ax3 = Axis(fig[2, 1]; aspect=DataAspect())
    ax4 = Axis(fig[2, 2]; aspect=DataAspect())
    ax5 = Axis(fig[3, 1]; aspect=DataAspect())
    ax6 = Axis(fig[3, 2]; aspect=DataAspect())

    surface!(ax1, x, y, bed; colormap=:turbo)
    surface!(ax1, x, y, surf2; colormap=:turbo)
    plot!(ax2, x, bed[:, sly])
    plot!(ax2, x, surf2[:, sly])
    heatmap!(ax3, x, z, wt.na.c[:, sly, :]; colormap=:turbo)
    heatmap!(ax4, x, z, wt.ns.c[:, sly, :]; colormap=:turbo)
    heatmap!(ax5, x, y, wt.na.c[slx, :, :]; colormap=:turbo)
    heatmap!(ax6, x, y, wt.ns.c[slx, :, :]; colormap=:turbo)

    return display(fig)
end

visme(synth_dem, Ψ, wt)
