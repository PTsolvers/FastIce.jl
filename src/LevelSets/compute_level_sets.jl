@kernel inbounds = true function _init_level_set!(Ψ::Field, dem::Field, dem_grid::StructuredGrid, Ψ_grid::StructuredGrid, cutoff, R)
    I = @index(Global, Cartesian)
    x, y, z = coord(Ψ_grid, location(Ψ), I)
    P = R * Point3(x, y, z)
    ud, sgn = sd_dem(P, cutoff, dem, dem_grid)
    Ψ[I] = ud * sgn
end

"""
    compute_level_set_from_dem!(arch, Ψ, dem, dem_grid, Ψ_grid, R[=LinearAlgebra.I])

Compute a level set from a given dem.
"""
function compute_level_set_from_dem!(arch::Architecture, Ψ::Field, dem::Field, dem_grid::CartesianGrid, Ψ_grid::CartesianGrid,
                                     R=LinearAlgebra.I)
    cutoff = 4maximum(spacing(Ψ_grid))
    kernel = _init_level_set!(backend(arch), 256, size(Ψ))
    kernel(Ψ, dem, dem_grid, Ψ_grid, cutoff, R)
    return
end
