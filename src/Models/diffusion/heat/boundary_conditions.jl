struct Value end
struct Flux end

struct BoundaryCondition{Kind, V}
    condition::V
    BoundaryCondition{Kind}(condition::V) where {Kind,V} = new{Kind,V}(condition)
end

const _COORDINATES  = ((:x, 1), (:y, 2), (:z, 3))

for (dim, val) in _COORDINATES
    ex_val = Expr(:(=), :T, :(DirichletBC{HalfCell}(bc.condition)))
    ex_val = Expr(:tuple, ex_val)

    ex_flux = Expr(:(=), Symbol(:q, dim), :(DirichletBC{FullCell}(bc.condition)))
    ex_flux = Expr(:tuple, ex_flux)

    @eval begin
        function extract_boundary_conditions(::Val{$val}, bc::BoundaryCondition{Value})
            return $ex_val, NamedTuple()
        end

        function extract_boundary_conditions(::Val{$val}, bc::BoundaryCondition{Flux})
            return NamedTuple(), $ex_flux
        end
    end
end

@inline no_bcs(names) = NamedTuple(f => nothing for f in names)

unique_names(a, b) = Tuple(unique(tuple(a..., b...)))

function expand_boundary_conditions(left, right)
    names = unique_names(keys(left), keys(right))

    if isempty(names)
        return nothing
    end

    default = no_bcs(names)
    left    = merge(default, left)
    right   = merge(default, right)

    return NamedTuple{names}(zip(left, right))
end

function make_field_boundary_conditions(bcs)
    ordering = (
        (:west  , :east),
        (:south , :north),
        (:bottom, :top),
    )

    field_bcs = ntuple(Val(length(ordering))) do dim
        left, right = bcs[ordering[dim]]

        left  = extract_boundary_conditions(Val(dim), left)
        right = extract_boundary_conditions(Val(dim), right)

        value, flux = Tuple(zip(left, right))

        value = expand_boundary_conditions(value...)
        flux  = expand_boundary_conditions(flux...)

        value, flux
    end

    return NamedTuple{(:value, :flux)}(zip(field_bcs...))
end

function _apply_bcs!(backend, grid, fields, bcs)
    field_map = (T = fields.T,
             qx = fields.q.x ,  qy = fields.q.y ,  qz = fields.q.z)

    ntuple(Val(length(bcs))) do D
        if !isnothing(bcs[D])
            fs = Tuple( field_map[f] for f in eachindex(bcs[D]) )
            apply_bcs!(Val(D), backend, grid, fs, values(bcs[D]))
        end
    end

    return
end
