"""
    init_levelset!(Ψ::Field, dem::AbstractField, dem_grid::UniformGrid, Ψ_grid::UniformGrid, cutoff, R)

Initialize level sets.
"""
@kernel inbounds = true function init_levelset!(Ψ::Field, dem::AbstractField, dem_grid::UniformGrid, Ψ_grid::UniformGrid, cutoff, R, O)
    I = @index(Global, NTuple)
    I = I + O
    x, y, z = coord(Ψ_grid, location(Ψ), I...)
    P = R * Point3(x, y, z)
    ud, sgn = sd_dem(P, cutoff, dem, dem_grid)
    Ψ[I...] = ud * sgn
end

@kernel inbounds = true function invert_levelset!(Ψ::Field, O)
    I = @index(Global, NTuple)
    I = I + O
    Ψ[I...] *= -1.0
end

"""
    compute_levelset_from_dem!(arch::Architecture, Ψ::Field, dem::AbstractField, dem_grid2D::UniformGrid, grid::UniformGrid, R=LinearAlgebra.I)

Compute level sets from dem.
"""
function compute_levelset_from_dem!(arch::Architecture, Ψ::Field, dem::AbstractField, dem_grid2D::UniformGrid, grid::UniformGrid, R=LinearAlgebra.I; outer_width=(16, 8, 4))
    cutoff = 4maximum(spacing(grid))
    launch = Launcher(arch, grid; outer_width=outer_width)
    launch(arch, grid, init_levelset! => (Ψ, dem, dem_grid2D, grid, cutoff, R); bc=batch(grid, Ψ => Neumann(); exchange=Ψ))
    return
end

"""
    invert_levelset!(arch::Architecture, Ψ::Field, grid::UniformGrid)

Invert level set `Ψ` to set what's below the surface as "inside".
"""
function invert_levelset!(arch::Architecture, Ψ::Field, grid::UniformGrid)
    launch = Launcher(arch, grid)
    launch(arch, grid, invert_levelset! => (Ψ,))
    return
end
