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
function compute_level_set_from_dem!(Ψ, dem, dem_grid, Ψ_grid)
    dx, dy, dz = spacing(Ψ_grid)
    cutoff = 4max(dx, dy, dz)
    R = LinearAlgebra.I
    nx, ny, nz = size(Ψ)
    _init_level_set!(Ψ, dem, dem_grid, Ψ_grid, cutoff, R)
    return
end
