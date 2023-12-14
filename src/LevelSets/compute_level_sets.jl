using FastIce
using FastIce.Grids
using KernelAbstractions

"Initialize level sets."
@kernel function _init_level_set!(Ψ, dem, dem_grid, Ψ_grid, cutoff, R)
    I = @index(Global, Cartesian)
    x, y, z = coord(Ψ_grid, location(Ψ), I)
    P = R * Point3(x, y, z)
    ud, sgn = sd_dem(P, cutoff, dem, dem_grid)
    Ψ[I] = ud * sgn
end

"Compute level sets from dem."
function compute_level_set_from_dem!(arch, Ψ, dem, dem_grid, Ψ_grid, R=LinearAlgebra.I)
    cutoff = 4maximum(spacing(Ψ_grid))
    _init_level_set!(backend(arch), 256, size(Ψ))(Ψ, dem, dem_grid, Ψ_grid, cutoff, R)
    return
end
