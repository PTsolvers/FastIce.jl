using Chmy.Architectures
using Chmy.Grids
using Chmy.Fields

using FastIce.LevelSets

using KernelAbstractions

using CUDA
using JLD2

# Data
# vavilov_path = "../data/vavilov.jld2"
# vavilov_path = "../data/Vavilov/Vavilov_80m.jld2"
vavilov_path = "../data/Vavilov/Vavilov_500m.jld2"
# synthetic_data = "../data/Synthetic/dome_glacier.jld2"
# synthetic_data = "../data/Synthetic/low_res_dome_glacier.jld2"

# Select backend
backend = CPU()
# backend = CUDABackend()
arch = Arch(backend)

"""
    load_dem(backend, arch::Architecture, path::String)

Load digital elevation map of surface and bedrock from (jld2) file, set dimensions of simulation,
initiate grids, copy data on gpu.
"""
function load_dem(backend, arch::Architecture, path::String)
    data = load(path)
    z_surf = data["DataElevation"].z_surf
    z_bed = data["DataElevation"].z_bed
    x = data["DataElevation"].x
    y = data["DataElevation"].y
    nx = length(x) - 1
    ny = length(y) - 1
    nz = 100
    # TODO: choose nz accordingly
    surf_lz = maximum(z_surf) - minimum(z_surf)
    bed_lz = maximum(z_bed) - minimum(z_bed)
    surf_Ψ_grid = UniformGrid(arch; origin=(0.0, 0.0, minimum(z_surf)), extent=(surf_lz, surf_lz, surf_lz), dims=(nx, ny, nz))
    bed_Ψ_grid = UniformGrid(arch; origin=(0.0, 0.0, minimum(z_bed)), extent=(bed_lz, bed_lz, bed_lz), dims=(nx, ny, nz))
    surf_dem_grid = UniformGrid(arch; origin=(0.0, 0.0), extent=(surf_lz, surf_lz), dims=(nx, ny))
    bed_dem_grid = UniformGrid(arch; origin=(0.0, 0.0), extent=(bed_lz, bed_lz), dims=(nx, ny))
    surf_field = Field(backend, surf_dem_grid, Vertex())
    bed_field = Field(backend, bed_dem_grid, Vertex())
    set!(surf_field, z_surf)
    set!(bed_field, z_bed)
    return surf_field, bed_field, surf_dem_grid, bed_dem_grid, surf_Ψ_grid, bed_Ψ_grid
end

"""
    load_synth_dem(backend, arch::Architecture, synthetic_data::String)

Load digital elevation map of surface and bedrock from (jld2) file, set dimensions of simulation,
initiate grids, copy data on gpu.
"""
function load_synth_dem(backend, arch::Architecture, synthetic_data::String)
    data = load(synthetic_data)
    z_surf = data["SyntheticElevation"].z_surf
    z_bed = data["SyntheticElevation"].z_bed
    nx = size(z_surf)[1] - 1
    ny = size(z_surf)[2] - 1
    nz = 10
    lz = maximum(z_surf) - minimum(z_surf)
    Ψ_grid = UniformGrid(arch; origin=(0.0, 0.0, minimum(z_surf)), extent=(lz, lz, lz), dims=(nx, ny, nz))
    dem_grid = UniformGrid(arch; origin=(0.0, 0.0), extent=(lz, lz), dims=(nx, ny))
    dem_bed = Field(backend, dem_grid, Vertex())
    dem_surf = Field(backend, dem_grid, Vertex())
    set!(dem_bed, z_bed)
    set!(dem_surf, z_surf)
    return dem_surf, dem_bed, dem_grid, Ψ_grid
end

# dem_surf, dem_bed, dem_grid, Ψ_grid = load_synth_dem(backend, arch, synthetic_data);
surf_field, bed_field, surf_dem_grid, bed_dem_grid, surf_Ψ_grid, bed_Ψ_grid = load_dem(backend, arch, vavilov_path);

Ψ = (
    na=Field(backend, surf_Ψ_grid, Vertex()),
    ns=Field(backend, bed_Ψ_grid, Vertex()),
)

compute_level_set_from_dem!(backend, Ψ[1], surf_field, surf_dem_grid, surf_Ψ_grid)
compute_level_set_from_dem!(backend, Ψ[2], bed_field, bed_dem_grid, bed_Ψ_grid)

# for phase in eachindex(Ψ)
#     compute_level_set_from_dem!(backend, Ψ[phase], dem_surf, dem_grid, Ψ_grid)
# end

wt = (
    na=volfrac_field(backend, surf_Ψ_grid),
    ns=volfrac_field(backend, bed_Ψ_grid),
)

compute_volume_fractions_from_level_set!(backend, wt[1], Ψ[1], surf_Ψ_grid)
compute_volume_fractions_from_level_set!(backend, wt[2], Ψ[2], bed_Ψ_grid)


# for phase in eachindex(Ψ)
#     compute_volume_fractions_from_level_set!(backend, wt[phase], Ψ[phase], Ψ_grid)
# end

# Save
jldopen(vavilov_path, "a+") do file
    file["level_sets_na"] = Array(interior(Ψ[1]))
    file["level_sets_ns"] = Array(interior(Ψ[2]))
    file["volume_frac_na"] = Array.(interior.(Tuple(wt[1])))
    file["volume_frac_ns"] = Array.(interior.(Tuple(wt[2])))
end
