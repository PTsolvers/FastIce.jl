struct Traction end
struct Velocity end
struct Slip     end

struct BoundaryCondition{Kind, N, C}
    components::C
    BoundaryCondition{Kind}(components::Tuple) where {Kind} = new{Kind, length(components), typeof(components)}(components)
end

BoundaryCondition{Kind}(components...) where {Kind} = BoundaryCondition{Kind}(components)

const _COORDINATES  = ((:x, 1), (:y, 2), (:z, 3))

for (dim, val) in _COORDINATES
    td = remove_dim(Val(val), _COORDINATES)
    TN = ntuple(Val(length(td))) do I
        dim2, val2 = td[I]
        val2 < val ? Symbol(:τ, dim2, dim) : Symbol(:τ, dim, dim2)
    end
    N  = Symbol(:τ, dim, dim)

    ex1 = Expr(:(=), :Pr, :(DirichletBC{HalfCell}(bc.components[$val])))
    ex2 = Expr(:(=), N,   :(DirichletBC{HalfCell}(convert(eltype(bc.components[$val]), 0))))
    ex3 = ntuple(Val(length(TN))) do I
        Expr(:(=), TN[I], :(DirichletBC{FullCell}(bc.components[$(td[I][2])])))
    end

    ex_tr = Expr(:tuple, ex1, ex2, ex3...)

    ex_vel = ntuple(length(_COORDINATES)) do I
        kind = I == val ? :(FullCell) : :(HalfCell)
        Expr(:(=), Symbol(:V, _COORDINATES[I][1]), :(DirichletBC{$kind}(bc.components[$I])))
    end

    ex_vel = Expr(:tuple, ex_vel...)

    ex_slip_tr = Expr(:tuple, ex3...)

    ex_slip_vel = Expr(:(=), Symbol(:V, dim), :(DirichletBC{FullCell}(bc.components[$val])))
    ex_slip_vel = Expr(:tuple, ex_slip_vel)

    @eval begin
        function extract_boundary_conditions(::Val{$val}, bc::BoundaryCondition{Traction})
            return $ex_tr, NamedTuple()
        end

        function extract_boundary_conditions(::Val{$val}, bc::BoundaryCondition{Velocity})
            return NamedTuple(), $ex_vel
        end

        function extract_boundary_conditions(::Val{$val}, bc::BoundaryCondition{Slip})
            return $ex_slip_tr, $ex_slip_vel
        end
    end
end

no_bcs(names) = NamedTuple(f => nothing for f in names)

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

        stress, velocity = Tuple(zip(left, right))

        stress   = expand_boundary_conditions(stress...)
        velocity = expand_boundary_conditions(velocity...)

        stress, velocity
    end

    return NamedTuple{(:stress, :velocity)}(zip(field_bcs...))
end

function _apply_bcs!(backend, grid, fields, bcs)
    field_map = (Pr = fields.Pr,
            τxx = fields.τ.xx, τyy = fields.τ.yy, τzz = fields.τ.zz,
            τxy = fields.τ.xy, τxz = fields.τ.xz, τyz = fields.τ.yz,
             Vx = fields.V.x ,  Vy = fields.V.y ,  Vz = fields.V.z)
    
    ntuple(Val(length(bcs))) do D
        if !isnothing(bcs[D])
            fs = Tuple( field_map[f] for f in eachindex(bcs[D]) )
            apply_bcs!(Val(D), backend, grid, fs, values(bcs[D]))
        end
    end

    return
end
