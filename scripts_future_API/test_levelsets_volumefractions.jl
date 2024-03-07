using Chmy.Architectures
using Chmy.Grids
using Chmy.Fields

using FastIce.LevelSets
using FastIce.Geometry

using KernelAbstractions
using JLD2

# using CUDA
# using AMDGPU

# Data
data_path = "./low_res_dome_rough_glacier.jld2"

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

dem_surf, dem_bed, dem_grid, Ψ_grid = extract_dem(arch, data_path)

Ψ = (
    na=Field(backend, Ψ_grid, Vertex()),
    ns=Field(backend, Ψ_grid, Vertex()),
)

for phase in eachindex(Ψ)
    compute_level_set_from_dem!(arch, Ψ[phase], dem_surf, dem_grid, Ψ_grid)
end

wt = (
    na=volfrac_field(backend, Ψ_grid),
    ns=volfrac_field(backend, Ψ_grid),
)

for phase in eachindex(Ψ)
    compute_volfrac_from_level_set!(arch, wt[phase], Ψ[phase], Ψ_grid)
end

# Save
save_path = "Vavilov_500m_out.jld2"
jldopen(vavilov_path, "a+") do file
    file["level_sets_na"] = Array(interior(Ψ[1]))
    file["level_sets_ns"] = Array(interior(Ψ[2]))
    file["volume_frac_na"] = Array.(interior.(Tuple(wt[1])))
    file["volume_frac_ns"] = Array.(interior.(Tuple(wt[2])))
end
