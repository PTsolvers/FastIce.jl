module BoundaryConditions

export discrete_bcs_x!, DirichletBC, HalfCell, FullCell

using FastIce.Grids

using KernelAbstractions
using Adapt

@kernel function discrete_bcs_x!(grid, fields, west_bcs, east_bcs)
    iy, iz = @index(Global, NTuple)
    for ifield in eachindex(fields) 
        apply_west_bc!(grid, fields[ifield], iy + 1, iz + 1, west_bcs[ifield])
        apply_east_bc!(grid, fields[ifield], iy + 1, iz + 1, east_bcs[ifield])
    end
end

struct HalfCell end
struct FullCell end

struct DirichletBC{FluxReconstruction, T}
    val::T
end

DirichletBC{FluxReconstruction}(val::T) where {FluxReconstruction, T} = DirichletBC{FluxReconstruction, T}(val)
(bc::DirichletBC{FR, <:Number})(grid, i, j) where FR = bc.val
(bc::DirichletBC{FR, <:AbstractArray})(grid, i, j) where FR = bc.val[i, j]


@inline get_flux(Δ, f2, bc::DirichletBC{HalfCell}, grid, i, j) = (bc(grid, i, j) - f2)/(0.5Δ)
@inline get_flux(Δ, f2, bc::DirichletBC{FullCell}, grid, i, j) = (bc(grid, i, j) - f2)/Δ

@inline function apply_west_bc!(grid, f, iy, iz, bc::DirichletBC)
    f[1, iy, iz] = f[2, iy, iz] + Δ(grid, 1) * get_flux(Δ(grid, 1), f[2, iy, iz], bc, grid, iy, iz)
end

@inline function apply_east_bc!(grid, f, iy, iz, bc::DirichletBC)
    f[end, iy, iz] = f[end-1, iy, iz] + Δ(grid, 1) * get_flux(Δ(grid, 1), f[end-1, iy, iz], bc, grid, iy, iz)
end

# @kernel function discrete_bcs_y!(f, grid, south_iy, north_iy, south_bc, north_bc)
#     ix, iz = @index(Global, NTuple)
#     apply_south_bc!(f, grid, ix + 1, south_iy, iz + 1, south_bc)
#     apply_north_bc!(f, grid, ix + 1, north_iy, iz + 1, north_bc)
# end

# @kernel function discrete_bcs_z!(f, bot_iz, top_iz, bot_bc, top_bc)
#     ix, iy = @index(Global, NTuple)
#     apply_bot_bc!(f, grid, ix + 1, iy + 1, bot_iz, bot_bc)
#     apply_top_bc!(f, grid, ix + 1, iy + 1, top_iz, top_bc)
# end

# struct NoBC end

# @inline apply_west_bc!(f, grid, ix, iy, iz, ::NoBC) = nothing
# @inline apply_east_bc!(f, grid, ix, iy, iz, ::NoBC) = nothing

# @inline apply_south_bc!(f, grid, ix, iy, iz, ::NoBC) = nothing
# @inline apply_north_bc!(f, grid, ix, iy, iz, ::NoBC) = nothing

# @inline apply_bot_bc!(f, grid, ix, iy, iz, ::NoBC) = nothing
# @inline apply_top_bc!(f, grid, ix, iy, iz, ::NoBC) = nothing

# struct Shift end
# struct NoShift end

# struct DirichletBC{T,HasShift}
#     val::T
# end

# DirichletBC{HasShift}(val::T) where {T,HasShift} = DirichletBC{T,HasShift}(val)

# (bc::DirichletBC{<:Number})(grid, i, j) = bc.val
# (bc::DirichletBC{<:AbstractArray})(grid, i, j) = @inbounds bc.val[i, j]

# Adapt.adapt_structure(to, f::DirichletBC) = Adapt.adapt(to, f.val)

# struct NeumannBC{T}
#     val::T
# end

# (bc::NeumannBC{<:Number})(grid, i, j) = bc.val
# (bc::NeumannBC{<:AbstractArray})(grid, i, j) = @inbounds bc.val[i, j]

# Adapt.adapt_structure(to, f::NeumannBC) = Adapt.adapt(to, f.val)

# @inline bc1!(f, I, v) = @inbounds f[I] = v
# @inline bc2!(f, I1, I2, a, b) = @inbounds f[I1] = a + b * f[I2]

# @inline apply_west_bc!(f, grid, ix, iy, iz, bc::DirichletBC{T,NoShift}) where {T} = bc1!(f, CartesianIndex(ix, iy, iz), bc(grid, iy, iz))
# @inline apply_east_bc!(f, grid, ix, iy, iz, bc::DirichletBC{T,NoShift}) where {T} = bc1!(f, CartesianIndex(ix, iy, iz), bc(grid, iy, iz))

# @inline apply_south_bc!(f, grid, ix, iy, iz, bc::DirichletBC{T,NoShift}) where {T} = bc1!(f, CartesianIndex(ix, iy, iz), bc(grid, ix, iz))
# @inline apply_north_bc!(f, grid, ix, iy, iz, bc::DirichletBC{T,NoShift}) where {T} = bc1!(f, CartesianIndex(ix, iy, iz), bc(grid, ix, iz))

# @inline apply_bot_bc!(f, grid, ix, iy, iz, bc::DirichletBC{T,NoShift}) where {T} = bc1!(f, CartesianIndex(ix, iy, iz), bc(grid, ix, iy))
# @inline apply_top_bc!(f, grid, ix, iy, iz, bc::DirichletBC{T,NoShift}) where {T} = bc1!(f, CartesianIndex(ix, iy, iz), bc(grid, ix, iy))

# @inline function apply_west_bc!(f, grid, ix, iy, iz, bc::DirichletBC{T,Shift}) where {T}
#     I1, I2 = CartesianIndex(ix, iy, iz), CartesianIndex(ix + 1, iy, iz)
#     bc2!(f, I1, I2, 2.0 * bc(grid, iy, iz), -1.0)
#     return
# end

# @inline function apply_east_bc!(f, grid, ix, iy, iz, bc::DirichletBC{T,Shift}) where {T}
#     I1, I2 = CartesianIndex(ix, iy, iz), CartesianIndex(ix - 1, iy, iz)
#     bc2!(f, I1, I2, 2.0 * bc(grid, iy, iz), -1.0)
#     return
# end

# @inline function apply_south_bc!(f, grid, ix, iy, iz, bc::DirichletBC{T,Shift}) where {T}
#     I1, I2 = CartesianIndex(ix, iy, iz), CartesianIndex(ix, iy + 1, iz)
#     bc2!(f, I1, I2, 2.0 * bc(grid, ix, iz), -1.0)
#     return
# end

# @inline function apply_north_bc!(f, grid, ix, iy, iz, bc::DirichletBC{T,Shift}) where {T}
#     I1, I2 = CartesianIndex(ix, iy, iz), CartesianIndex(ix, iy - 1, iz)
#     bc2!(f, I1, I2, 2.0 * bc(grid, ix, iz), -1.0)
#     return
# end

# @inline function apply_bot_bc!(f, grid, ix, iy, iz, bc::DirichletBC{T,Shift}) where {T}
#     I1, I2 = CartesianIndex(ix, iy, iz), CartesianIndex(ix, iy, iz + 1)
#     bc2!(f, I1, I2, 2.0 * bc(grid, ix, iy), -1.0)
#     return
# end

# @inline function apply_top_bc!(f, grid, ix, iy, iz, bc::DirichletBC{T,Shift}) where {T}
#     I1, I2 = CartesianIndex(ix, iy, iz), CartesianIndex(ix, iy, iz - 1)
#     bc2!(f, I1, I2, 2.0 * bc(grid, ix, iy), -1.0)
#     return
# end

# @inline function apply_west_bc!(f, grid, ix, iy, iz, bc::NeumannBC)
#     I1, I2 = CartesianIndex(ix, iy, iz), CartesianIndex(ix + 1, iy, iz)
#     bc2!(f, I1, I2, -Δ(grid, 1) * bc(grid, iy, iz), 1.0)
#     return
# end

# @inline function apply_east_bc!(f, grid, ix, iy, iz, bc::NeumannBC)
#     I1, I2 = CartesianIndex(ix, iy, iz), CartesianIndex(ix - 1, iy, iz)
#     bc2!(f, I1, I2, Δ(grid, 1) * bc(grid, iy, iz), 1.0)
#     return
# end

# @inline function apply_south_bc!(f, grid, ix, iy, iz, bc::NeumannBC)
#     I1, I2 = CartesianIndex(ix, iy, iz), CartesianIndex(ix, iy + 1, iz)
#     bc2!(f, I1, I2, -Δ(grid, 2) * bc(grid, ix, iz), 1.0)
#     return
# end

# @inline function apply_north_bc!(f, grid, ix, iy, iz, bc::NeumannBC)
#     I1, I2 = CartesianIndex(ix, iy, iz), CartesianIndex(ix, iy - 1, iz)
#     bc2!(f, I1, I2, Δ(grid, 2) * bc(grid, ix, iz), 1.0)
#     return
# end

# @inline function apply_bot_bc!(f, grid, ix, iy, iz, bc::NeumannBC)
#     I1, I2 = CartesianIndex(ix, iy, iz), CartesianIndex(ix, iy, iz + 1)
#     bc2!(f, I1, I2, -Δ(grid, 3) * bc(grid, ix, iy), 1.0)
#     return
# end

# @inline function apply_top_bc!(f, grid, ix, iy, iz, bc::NeumannBC)
#     I1, I2 = CartesianIndex(ix, iy, iz), CartesianIndex(ix, iy, iz - 1)
#     bc2!(f, I1, I2, Δ(grid, 3) * bc(grid, ix, iy), 1.0)
#     return
# end

end