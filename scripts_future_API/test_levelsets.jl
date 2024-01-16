using CUDA
using FileIO
using FastIceTools
using JLD2
using KernelAbstractions
using HDF5

using FastIce.Grids
using FastIce.Fields
using FastIce.LevelSets
using FastIce.Architectures
using FastIce.Writers

# vavilov_path = "../data/Vavilov/vavilov.jld2"
synthetic_data = "../data/synthetic.jld2"

# Select backend (CPU(), CUDABackend())
backend = CUDABackend()
arch = Architecture(backend)

"""
    load_dem_on_GPU(path::String, arch::Architecture)

Load digital elevation map of surface and bedrock from (jld2) file, set dimensions of simulation, 
initiate grids, copy data on gpu.
"""
function load_dem_on_GPU(path::String, arch::Architecture)
    data     = load(path)
    x        = data["DataElevation"].x
    y        = data["DataElevation"].y
    R        = data["DataElevation"].rotation
    z_surf   = permutedims(data["DataElevation"].z_surf)
    z_bed    = permutedims(data["DataElevation"].z_bed)
    domain   = data["DataElevation"].domain
    nx       = length(x) - 1
    ny       = length(y) - 1
    nz       = 10
    z        = LinRange(domain.zmin, domain.zmax, nz)
    Ψ_grid   = CartesianGrid(; origin=(0.0, 0.0, 0.0), extent=(1.0, 1.0, 1.0), size=(nx, ny, nz))
    Ψ        = Field(arch, Ψ_grid, Vertex())
    dem_grid = CartesianGrid(; origin=(0.0, 0.0), extent=(1.0, 1.0), size=(nx, ny))
    dem_bed  = Field(arch, dem_grid, Vertex())
    dem_surf = Field(arch, dem_grid, Vertex())
    set!(dem_bed, z_bed)
    set!(dem_surf, z_surf)
    return Ψ, dem_surf, dem_grid, Ψ_grid
end

function load_synth_dem_on_GPU(path::String, arch::Architecture)
    data   = load(path)
    z_surf = data["z_surf"]
    z_bed  = data["z_bed"]
    nx     = size(z_bed)[1] - 1
    ny     = size(z_bed)[2] - 1
    nz     = 100
    # z        = LinRange(minimum(z_bed), maximum(z_surf), nz)
    Ψ_grid   = CartesianGrid(; origin=(0.0, 0.0, 0.0), extent=(1.0, 1.0, 1.0), size=(nx, ny, nz))
    Ψ        = Field(arch, Ψ_grid, Vertex())
    dem_grid = CartesianGrid(; origin=(0.0, 0.0), extent=(1.0, 1.0), size=(nx, ny))
    dem_bed  = Field(arch, dem_grid, Vertex())
    dem_surf = Field(arch, dem_grid, Vertex())
    set!(dem_bed, z_bed)
    set!(dem_surf, z_surf)
    return Ψ, dem_surf, dem_bed, dem_grid, Ψ_grid
end

# Ψ, dem_surf, dem_grid, Ψ_grid = load_dem_on_GPU(vavilov_path, arch);
Ψ, _, dem_bed, dem_grid, Ψ_grid = load_synth_dem_on_GPU(synthetic_data, arch);

compute_level_set_from_dem!(arch, Ψ, dem_bed, dem_grid, Ψ_grid)

jldopen(synthetic_data, "a+") do file
    file["level_sets"] = Array(Ψ)
    file["dem_bed"] = Array(dem_bed)
    file["dem_surf"] = Array(dem_surf)
end