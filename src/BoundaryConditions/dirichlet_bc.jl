# Reconstruct gradient from the interface between two grid locations
struct HalfCell end

# Reconstruct gradient from the ghost node location
struct FullCell end

# First-kind Dirichlet boundary conditon parametrised by the gradient reconstruction kind (can be HalfCell or FullCell)
struct DirichletBC{Gradient,T} <: FieldBoundaryCondition
    condition::T
    DirichletBC{Gradient}(condition::T) where {Gradient,T} = new{Gradient,T}(condition)
end

# Create a DirichletBC with a continuous or discrete boundary function
function DirichletBC{G}(fun::Function; kwargs...) where {G}
    condition = BoundaryFunction(fun; kwargs...)
    return DirichletBC{G}(condition)
end

@inline (bc::DirichletBC{G,<:Number})(grid, loc, dim, I) where {G} = bc.condition
Base.@propagate_inbounds (bc::DirichletBC{G,<:AbstractArray})(grid, loc, dim, I) where {G} = bc.condition[remove_dim(dim, I)]

Base.@propagate_inbounds (bc::DirichletBC{G,<:BoundaryFunction})(grid, loc, dim, I) where {G} = bc.condition(grid, loc, dim, I)

Adapt.adapt_structure(to, f::DirichletBC{G,<:AbstractArray}) where {G} = DirichletBC{G}(Adapt.adapt(to, parent(f.condition)))

Base.@propagate_inbounds _get_gradient(f2, bc::DirichletBC{HalfCell}, grid, loc, dim, I) = 2 * (bc(grid, loc, dim, I) - f2)
Base.@propagate_inbounds _get_gradient(f2, bc::DirichletBC{FullCell}, grid, loc, dim, I) = (bc(grid, loc, dim, I) - f2)

@inline function _apply_field_boundary_condition!(side, dim, grid, f, loc, Ibc, bc::DirichletBC)
    I = _bc_index(dim, side, loc, size(f), Ibc)
    DI = _bc_offset(Val(ndims(grid)), dim, side)
    @inbounds f[I] = f[I+DI] + _get_gradient(f[I+DI], bc, grid, loc, dim, I)
    return
end
