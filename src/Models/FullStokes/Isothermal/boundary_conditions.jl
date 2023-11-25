struct Traction end
struct Velocity end
struct Slip end

"""
Boundary condition for Stokes problem.
"""
struct BoundaryCondition{Kind,N,C}
    components::C
    BoundaryCondition{Kind}(components::Tuple) where {Kind} = new{Kind,length(components),typeof(components)}(components)
end

BoundaryCondition{Kind}(components...) where {Kind} = BoundaryCondition{Kind}(components)

function instantiate_boundary_conditions(coords::NTuple{ND}) where {ND}
    for (dim, val) in coords
        td = remove_dim(Val(val), coords)
        TN = ntuple(Val(length(td))) do I
            dim2, val2 = td[I]
            val2 < val ? Symbol(:τ, dim2, dim) : Symbol(:τ, dim, dim2)
        end
        N  = Symbol(:τ, dim, dim)
    
        ex1 = Expr(:(=), :Pr, :(DirichletBC{HalfCell}(bc.components[$val])))
        ex2 = Expr(:(=), N, :(DirichletBC{HalfCell}(convert(eltype(bc.components[$val]), 0))))
        ex3 = ntuple(Val(length(TN))) do I
            Expr(:(=), TN[I], :(DirichletBC{FullCell}(bc.components[$(td[I][2])])))
        end
    
        ex_tr = Expr(:tuple, ex1, ex2, ex3...)
    
        ex_vel = ntuple(length(coords)) do I
            kind = I == val ? :(FullCell) : :(HalfCell)
            Expr(:(=), Symbol(:V, coords[I][1]), :(DirichletBC{$kind}(bc.components[$I])))
        end
    
        ex_vel = Expr(:tuple, ex_vel...)
    
        ex_slip_tr = Expr(:tuple, ex3...)
    
        ex_slip_vel = Expr(:(=), Symbol(:V, dim), :(DirichletBC{FullCell}(bc.components[$val])))
        ex_slip_vel = Expr(:tuple, ex_slip_vel)
    
        @eval begin
            extract_stress_bc(::Val{$val}, bc::BoundaryCondition{Traction,$ND}) = $ex_tr
            extract_stress_bc(::Val{$val}, bc::BoundaryCondition{Velocity,$ND}) = ()
            extract_stress_bc(::Val{$val}, bc::BoundaryCondition{Slip,$ND})     = $ex_slip_tr
    
            extract_velocity_bc(::Val{$val}, bc::BoundaryCondition{Traction,$ND}) = ()
            extract_velocity_bc(::Val{$val}, bc::BoundaryCondition{Velocity,$ND}) = $ex_vel
            extract_velocity_bc(::Val{$val}, bc::BoundaryCondition{Slip,$ND})     = $ex_slip_vel
        end
    end
end

instantiate_boundary_conditions(((:x, 1), (:y, 2)))
instantiate_boundary_conditions(((:x, 1), (:y, 2), (:z, 3)))

make_batch(::Tuple{}, fields) = nothing

function make_batch(bcs::NamedTuple, fields::NamedTuple)
    batch_fields = Tuple(fields[name] for name in eachindex(bcs))
    return BoundaryConditionsBatch(batch_fields, values(bcs))
end

@inline function dim_side_ntuple(f::F, ::Val{N}) where {F,N}
    ntuple(D -> ntuple(S -> f(D, S), Val(2)), Val(N))
end

function make_stress_bc(arch::Architecture{Kind}, ::CartesianGrid{N}, fields, bc) where {Kind,N}
    ordering = (:x, :y, :z)
    dim_side_ntuple(Val(N)) do D, S
        if (Kind == Distributed.DistributedMPI) && has_neighbor(details(arch), D, S)
            nothing
        else
            new_bc = extract_stress_bc(Val(D), bc[ordering[D]][S])
            make_batch(new_bc, fields)
        end
    end
end

function make_velocity_bc(arch::Architecture{Kind}, ::CartesianGrid{N}, fields::NamedTuple{names}, bc) where {Kind,N,names}
    ordering = (:x, :y, :z)
    dim_side_ntuple(Val(N)) do D, S
        if (Kind == Distributed.DistributedMPI) && has_neighbor(details(arch), D, S)
            new_bc = NamedTuple{names}(ExchangeInfo(Val(S), Val(D), V) for V in fields)
            make_batch(new_bc, fields)
        else
            new_bc = extract_velocity_bc(Val(D), bc[ordering[D]][S])
            make_batch(new_bc, fields)
        end
    end
end

function make_residuals_bc(arch::Architecture{Kind}, ::CartesianGrid{N}, fields::NamedTuple{names}, bc) where {Kind,N,names}
    ordering = (:x, :y, :z)
    dim_side_ntuple(Val(N)) do D, S
        if (Kind == Distributed.DistributedMPI) && has_neighbor(details(arch), D, S)
            nothing
        else
            if !(bc[ordering[D]][S] isa BoundaryCondition{Traction})
                Vn = Symbol(:r_V, ordering[D])
                BoundaryConditionsBatch((fields[Vn],), (DirichletBC{FullCell}(0.0),))
            else
                nothing
            end
        end
    end
end

function make_rheology_bc(arch::Architecture{Kind}, ::CartesianGrid{N}, η) where {Kind,N}
    dim_side_ntuple(Val(N)) do D, S
        if (Kind == Distributed.DistributedMPI) && has_neighbor(details(arch), D, S)
            BoundaryConditionsBatch((η,), (ExchangeInfo(Val(S), Val(D), η),))
        else
            BoundaryConditionsBatch((η,), (ExtrapolateBC(),))
        end
    end
end

function make_field_boundary_conditions(arch::Architecture, grid::CartesianGrid, fields, bc)
    stress_fields   = (; Pr=fields.Pr, NamedTuple{Symbol.(:τ, keys(fields.τ))}(values(fields.τ))...)
    velocity_fields = NamedTuple{Symbol.(:V, keys(fields.V))}(values(fields.V))
    residual_fields = NamedTuple{Symbol.(:r_V, keys(fields.r_V))}(values(fields.r_V))

    return (stress   = make_stress_bc(arch, grid, stress_fields, bc),
            velocity = make_velocity_bc(arch, grid, velocity_fields, bc),
            rheology = make_rheology_bc(arch, grid, fields.η),
            residual = make_residuals_bc(arch, grid, residual_fields, bc))
end
