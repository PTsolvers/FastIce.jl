using FastIce
using FastIce.Grids
using KernelAbstractions

"Initialize level sets."
@kernel function _init_level_set!(Ψ, dem, dem_grid, Ψ_grid, cutoff, R)
    ix, iy, iz = @index(Global, NTuple)
    x, y, z = Ψ_grid[1][ix], Ψ_grid[2][iy], Ψ_grid[3][iz]
    P = R * Point3(x, y, z)
    ud, sgn = sd_dem(P, cutoff, dem, dem_grid)
    Ψ[ix, iy, iz] = ud * sgn
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
