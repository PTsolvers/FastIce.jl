
using FastIceTools
using FileIO
using JLD
using CUDA
using FastIce
using FastIce.Grids
using FastIce.LevelSets

vavilov_path = "../data/vavilov.jld"

function load_dem_on_GPU(path::String)
    data = load(path)
    x = data["DataElevation"].x
    y = data["DataElevation"].y
    R = data["DataElevation"].rotation
    z_surf = data["DataElevation"].z_surf
    z_bed = data["DataElevation"].z_bed
    domain = data["DataElevation"].domain
    nx = length(x)
    ny = length(y)
    nz = 100
    z = LinRange(domain.zmin, domain.zmax, nz)
    Ψ        = CUDA.zeros(nx, ny, nz)
    Ψ_grid   = CartesianGrid(origin=(0.0,0.0,0.0), extent=(1.0,1.0,1.0), size=(nx, ny, nz))
    dem_bed  = CuArray(z_bed)
    dem_surf = CuArray(z_surf)
    dem_grid = CartesianGrid(origin=(0.0,0.0), extent=(1.0,1.0), size=(nx, ny))
    return Ψ, dem_surf, dem_grid, Ψ_grid
end

@time compute_level_set_from_dem!(load_dem_on_GPU(vavilov_path)...)
