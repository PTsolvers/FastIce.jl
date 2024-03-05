"""
    _init_level_set!(Ψ::Field, dem::Field, dem_grid::UniformGrid, Ψ_grid::UniformGrid, cutoff::AbstractFloat, R::AbstractMatrix)

Initialize level sets.
"""
@kernel function init_level_set!(Ψ::Field, dem::Field, dem_grid::UniformGrid, Ψ_grid::UniformGrid, cutoff, R)
    I = @index(Global, NTuple)
    x, y, z = coord(Ψ_grid, location(Ψ), I...)
    P = R * Point3(x, y, z)
    ud, sgn = sd_dem(P, cutoff, dem, dem_grid)
    Ψ[I...] = ud * sgn
end


"""
    compute_level_set_from_dem!(arch::Architecture, Ψ::Field, dem::Field, dem_grid::UniformGrid, Ψ_grid::UniformGrid, R=LinearAlgebra.I)

Compute level sets from dem.
"""
function compute_level_set_from_dem!(backend, Ψ::Field, dem::Field, dem_grid::UniformGrid, Ψ_grid::UniformGrid, R=LinearAlgebra.I)
    kernel = init_level_set!(backend, 256, size(Ψ))
    cutoff = 4maximum(spacing(Ψ_grid, Center()))
    kernel(Ψ, dem, dem_grid, Ψ_grid, cutoff, R)
    return
end