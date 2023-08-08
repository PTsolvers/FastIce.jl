module BoundaryConditions

export discrete_bcs_x!, discrete_bcs_y!, discrete_bcs_z!
export DirichletBC, HalfCell, FullCell, NoBC

using FastIce.Grids

using KernelAbstractions
using Adapt

@kernel function discrete_bcs_x!(grid, fields, west_bcs, east_bcs)
    iy, iz = @index(Global, NTuple)
    for ifield in eachindex(fields)
        apply_west_x_bc!(grid, fields[ifield], iy, iz, west_bcs[ifield])
        apply_east_x_bc!(grid, fields[ifield], iy, iz, east_bcs[ifield])
    end
end

@kernel function discrete_bcs_y!(grid, fields, south_bcs, north_bcs)
    ix, iz = @index(Global, NTuple)
    for ifield in eachindex(fields)
        apply_south_y_bc!(grid, fields[ifield], ix, iz, south_bcs[ifield])
        apply_north_y_bc!(grid, fields[ifield], ix, iz, north_bcs[ifield])
    end
end

@kernel function discrete_bcs_z!(grid, fields, bot_bcs, top_bcs)
    ix, iy = @index(Global, NTuple)
    for ifield in eachindex(fields)
        apply_bot_z_bc!(grid, fields[ifield], ix, iy, bot_bcs[ifield])
        apply_top_z_bc!(grid, fields[ifield], ix, iy, top_bcs[ifield])
    end
end

struct NoBC end

apply_west_x_bc!(grid, fields, iy, iz, ::NoBC) = nothing
apply_east_x_bc!(grid, fields, iy, iz, ::NoBC) = nothing

apply_south_y_bc!(grid, fields, ix, iz, ::NoBC) = nothing
apply_north_y_bc!(grid, fields, ix, iz, ::NoBC) = nothing

apply_bot_z_bc!(grid, fields, ix, iy, ::NoBC) = nothing
apply_top_z_bc!(grid, fields, ix, iy, ::NoBC) = nothing

struct HalfCell end
struct FullCell end

struct DirichletBC{FluxReconstruction, T}
    val::T
end

DirichletBC{FluxReconstruction}(val::T) where {FluxReconstruction, T} = DirichletBC{FluxReconstruction, T}(val)

(bc::DirichletBC{FR, <:Number})(grid, i, j) where FR = bc.val
Base.@propagate_inbounds (bc::DirichletBC{FR, <:AbstractArray})(grid, i, j) where FR = bc.val[i, j]

Adapt.adapt_structure(to, f::DirichletBC{FR, <:AbstractArray}) where FR = DirichletBC{FR}(Adapt.adapt(to, parent(f.val)))

Base.@propagate_inbounds get_flux(Δ, f2, bc::DirichletBC{HalfCell}, grid, i, j) = (bc(grid, i, j) - f2)/(0.5Δ)
Base.@propagate_inbounds get_flux(Δ, f2, bc::DirichletBC{FullCell}, grid, i, j) = (bc(grid, i, j) - f2)/Δ

@inline function apply_west_x_bc!(grid, f, iy, iz, bc::DirichletBC)
    @inbounds f[1, iy, iz] = f[2, iy, iz] + Δ(grid, 1) * get_flux(Δ(grid, 1), f[2, iy, iz], bc, grid, iy, iz)
end

@inline function apply_east_x_bc!(grid, f, iy, iz, bc::DirichletBC)
    @inbounds f[end, iy, iz] = f[end-1, iy, iz] + Δ(grid, 1) * get_flux(Δ(grid, 1), f[end-1, iy, iz], bc, grid, iy, iz)
end

@inline function apply_south_y_bc!(grid, f, ix, iz, bc::DirichletBC)
    @inbounds f[ix, 1, iz] = f[ix, 2, iz] + Δ(grid, 2) * get_flux(Δ(grid, 2), f[ix, 2, iz], bc, grid, ix, iz)
end

@inline function apply_north_y_bc!(grid, f, ix, iz, bc::DirichletBC)
    @inbounds f[ix, end, iz] = f[ix, end-1, iz] + Δ(grid, 2) * get_flux(Δ(grid, 2), f[ix, end-1, iz], bc, grid, ix, iz)
end

@inline function apply_bot_z_bc!(grid, f, ix, iy, bc::DirichletBC)
    @inbounds f[ix, iy, 1] = f[ix, iy, 2] + Δ(grid, 3) * get_flux(Δ(grid, 3), f[ix, iy, 2], bc, grid, ix, iy)
end

@inline function apply_top_z_bc!(grid, f, ix, iy, bc::DirichletBC)
    @inbounds f[ix, iy, end] = f[ix, iy, end-1] + Δ(grid, 3) * get_flux(Δ(grid, 3), f[ix, iy, end-1], bc, grid, ix, iy)
end

end