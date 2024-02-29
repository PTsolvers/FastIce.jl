# Boundary conditions that extrapolates the values of the field outside of the domain
struct ExtrapolateBC <: FieldBoundaryCondition end

@inline function _apply_field_boundary_condition!(side, dim, grid, f, loc, Ibc, ::ExtrapolateBC)
    I = _bc_index(dim, side, loc, size(f), Ibc)
    DI = _bc_offset(Val(ndims(grid)), dim, side)
    @inbounds f[I] = 2 * f[I+DI] - f[I+2DI]
    return
end

# Boundary conditions that extrapolates the values of the field outside of the domain in logarigthmic space
struct ExtrapolateLogBC <: FieldBoundaryCondition end

@inline function _apply_field_boundary_condition!(side, dim, grid, f, loc, Ibc, ::ExtrapolateLogBC)
    I = _bc_index(dim, side, loc, size(f), Ibc)
    DI = _bc_offset(Val(ndims(grid)), dim, side)
    @inbounds f[I] = exp(2 * log(f[I+DI]) - log(f[I+2DI]))
    return
end

# Boundary conditions that extrapolates the values of the field outside of the domain by copying them
struct ExpandBC <: FieldBoundaryCondition end

@inline function _apply_field_boundary_condition!(side, dim, grid, f, loc, Ibc, ::ExpandBC)
    I = _bc_index(dim, side, loc, size(f), Ibc)
    DI = _bc_offset(Val(ndims(grid)), dim, side)
    @inbounds f[I] = f[I+DI]
    return
end
