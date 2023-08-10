module Isothermal

export BoundaryCondition, Traction, Velocity, Slip
export IsothermalFullStokesModel, advance_iteration!, advance_timestep!

using FastIce.Physics
using FastIce.Grids
using FastIce.Fields
using FastIce.BoundaryConditions
using FastIce.Utils

include("kernels.jl")

function default_physics(::Type{T}) where T
    return (
        equation_of_state=default(IncompressibleIceEOS{T}),
        thermal_properties=default(IceThermalProperties{T}),
        rheology=default(GlensLawRheology{Int64})
    )
end

struct IsothermalFullStokesModel{Backend,Grid,BC,Physics,IterParams,Fields}
    backend::Backend
    grid::Grid
    boundary_conditions::BC
    physics::Physics
    iter_params::IterParams
    fields::Fields
end

function make_fields_mechanics(backend, grid)
    return (
        Pr = Field(backend, grid, Center(), (1, 1, 1)),
        τ  = (
            xx = Field(backend, grid, Center(), (1, 1, 1)),
            yy = Field(backend, grid, Center(), (1, 1, 1)),
            zz = Field(backend, grid, Center(), (1, 1, 1)),
            xy = Field(backend, grid, Vertex()),
            xz = Field(backend, grid, Vertex()),
            yz = Field(backend, grid, Vertex()),
        ),
        V = (
            x = Field(backend, grid, (Vertex(), Center(), Center()), (0, 1, 1)),
            y = Field(backend, grid, (Center(), Vertex(), Center()), (1, 0, 1)),
            z = Field(backend, grid, (Center(), Center(), Vertex()), (1, 1, 0)),
        )
    )
end

function IsothermalFullStokesModel(; backend, grid, boundary_conditions, physics=nothing, iter_params, fields=nothing, other_fields=nothing)
    if isnothing(fields)
        mechanic_fields = make_fields_mechanics(backend, grid)
        rheology_fields = (η = Field(backend, grid, Center(), (1, 1, 1)),)
        fields = merge(mechanic_fields, rheology_fields)
    end

    if !isnothing(other_fields)
        fields = merge(fields, other_fields)
    end

    if isnothing(physics)
        physics = default_physics(eltype(grid))
    end

    boundary_conditions = create_field_boundary_conditions(boundary_conditions)

    return IsothermalFullStokesModel(backend, grid, boundary_conditions, physics, iter_params, fields)
end

fields(model::IsothermalFullStokesModel) = model.fields
grid(model::IsothermalFullStokesModel) = model.grid

struct Traction end
struct Velocity end
struct Slip     end

struct BoundaryCondition{Kind, Tx, Ty, Tz}
    x::Tx
    y::Ty
    z::Tz
end

BoundaryCondition{Kind}(x::Tx, y::Ty, z::Tz) where {Kind, Tx, Ty, Tz} = BoundaryCondition{Kind, Tx, Ty, Tz}(x, y, z)

function extract_x_bcs(bc::BoundaryCondition{Traction})
    return (Pr  = DirichletBC{HalfCell}(bc.x),
            τxx = DirichletBC{HalfCell}(convert(eltype(bc.x), 0)),
            τxy = DirichletBC{FullCell}(bc.y),
            τxz = DirichletBC{FullCell}(bc.z),),
            NamedTuple()
end

function extract_y_bcs(bc::BoundaryCondition{Traction})
    return (Pr  = DirichletBC{HalfCell}(bc.y),
            τyy = DirichletBC{HalfCell}(convert(eltype(bc.y), 0)),
            τxy = DirichletBC{FullCell}(bc.x),
            τyz = DirichletBC{FullCell}(bc.z),),
            NamedTuple()
end

function extract_z_bcs(bc::BoundaryCondition{Traction})
    return (Pr  = DirichletBC{HalfCell}(bc.z),
            τyy = DirichletBC{HalfCell}(convert(eltype(bc.z), 0)),
            τxz = DirichletBC{FullCell}(bc.x),
            τyz = DirichletBC{FullCell}(bc.y),),
            NamedTuple()
end

function extract_x_bcs(bc::BoundaryCondition{Velocity})
    return NamedTuple(),
          (Vx = DirichletBC{FullCell}(bc.x),
           Vy = DirichletBC{HalfCell}(bc.y),
           Vz = DirichletBC{HalfCell}(bc.z),)
end

function extract_y_bcs(bc::BoundaryCondition{Velocity})
    return NamedTuple(),
          (Vx = DirichletBC{HalfCell}(bc.x),
           Vy = DirichletBC{FullCell}(bc.y),
           Vz = DirichletBC{HalfCell}(bc.z),)
end

function extract_z_bcs(bc::BoundaryCondition{Velocity})
    return NamedTuple(),
           (Vx = DirichletBC{HalfCell}(bc.x),
            Vy = DirichletBC{HalfCell}(bc.y),
            Vz = DirichletBC{FullCell}(bc.z),)
end

function extract_x_bcs(bc::BoundaryCondition{Slip})
    return (τxy = DirichletBC{FullCell}(bc.y),
            τxz = DirichletBC{FullCell}(bc.z),),
           ( Vx = DirichletBC{FullCell}(bc.x),)
end

function extract_y_bcs(bc::BoundaryCondition{Slip})
    return (τxy = DirichletBC{FullCell}(bc.x),
            τyz = DirichletBC{FullCell}(bc.z),),
           ( Vy = DirichletBC{FullCell}(bc.y),)
end

function extract_z_bcs(bc::BoundaryCondition{Slip})
    return (τxz = DirichletBC{FullCell}(bc.x),
            τyz = DirichletBC{FullCell}(bc.y),),
           ( Vz = DirichletBC{FullCell}(bc.z),)
end

@inline no_bcs(names) = NamedTuple(f => NoBC() for f in names)

unique_names(a, b) = Tuple(unique(tuple(a..., b...)))

function create_field_boundary_conditions(f, left, right)
    left_stress , left_velocity  = f(left)
    right_stress, right_velocity = f(right)

    stress_names   = unique_names(keys(left_stress)  , keys(right_stress))
    velocity_names = unique_names(keys(left_velocity), keys(right_velocity))

    default_stress   = no_bcs(stress_names)
    default_velocity = no_bcs(velocity_names)

    left_stress  = merge(default_stress, left_stress)
    right_stress = merge(default_stress, right_stress)

    left_velocity  = merge(default_velocity, left_velocity)
    right_velocity = merge(default_velocity, right_velocity)

    return stress_names, left_stress, right_stress, velocity_names, left_velocity, right_velocity
end

function create_field_boundary_conditions(bcs)
    stress_names_x, west_stress_bcs , east_stress_bcs , velocity_names_x, west_velocity_bcs , east_velocity_bcs  = create_field_boundary_conditions(extract_x_bcs, bcs.west, bcs.east)
    stress_names_y, south_stress_bcs, north_stress_bcs, velocity_names_y, south_velocity_bcs, north_velocity_bcs = create_field_boundary_conditions(extract_y_bcs, bcs.south, bcs.north)
    stress_names_z, bot_stress_bcs  , top_stress_bcs  , velocity_names_z, bot_velocity_bcs  , top_velocity_bcs   = create_field_boundary_conditions(extract_z_bcs, bcs.bot  , bcs.top)
    return (
        stress = (
            x = (
                names = stress_names_x,
                left  = west_stress_bcs,
                right = east_stress_bcs,
            ),
            y = (
                names = stress_names_y,
                left  = south_stress_bcs,
                right = north_stress_bcs,
            ),
            z = (
                names = stress_names_z,
                left  = bot_stress_bcs,
                right = top_stress_bcs,
            )
        ),
        velocity = (
            x = (
                names = velocity_names_x,
                left  = west_velocity_bcs,
                right = east_velocity_bcs,
            ),
            y = (
                names = velocity_names_y,
                left  = south_velocity_bcs,
                right = north_velocity_bcs,
            ),
            z = (
                names = velocity_names_z,
                left  = bot_velocity_bcs,
                right = top_velocity_bcs,
            )
        )
    )
end

function apply_bcs!(backend, grid, fields, bcs)
    field_map = (Pr = fields.Pr,
            τxx = fields.τ.xx, τyy = fields.τ.yy, τzz = fields.τ.zz,
            τxy = fields.τ.xy, τxz = fields.τ.xz, τyz = fields.τ.yz,
             Vx = fields.V.x ,  Vy = fields.V.y ,  Vz = fields.V.z)

    function apply_bcs_dim!(f, dim)
        fields    = Tuple( interior_and_halo(field_map[f], dim) for f in dim.names )
        left_bcs  = values(dim.left)
        right_bcs = values(dim.right)
        f(grid, fields, left_bcs, right_bcs)
    end
    
    apply_bcs_dim!(discrete_bcs_x!(backend, 256, (size(grid, 2)+1, size(grid, 3)+1)), bcs.x)
    apply_bcs_dim!(discrete_bcs_y!(backend, 256, (size(grid, 1)+1, size(grid, 3)+1)), bcs.y)
    apply_bcs_dim!(discrete_bcs_z!(backend, 256, (size(grid, 1)+1, size(grid, 2)+1)), bcs.z)

    return
end

function advance_iteration!(model::IsothermalFullStokesModel, t, Δt; async = true)
    (; Pr, τ, V, η) = model.fields
    (; η_rel, Δτ) = model.iter_params
    η_rh = model.physics.rheology
    Δ = NamedTuple{(:x, :y, :z)}(spacing(model.grid))
    nx, ny, nz = size(model.grid)
    backend = model.backend

    set_bcs!(bcs) = apply_bcs!(model.backend, model.grid, model.fields, bcs)

    # stress
    update_σ!(backend, 256, (nx, ny, nz))(interior(Pr), interior(τ), V, η, Δτ, Δ)
    set_bcs!(model.boundary_conditions.stress)
    # velocity
    update_V!(backend, 256, (nx + 1, ny + 1, nz + 1))(interior(V), Pr, τ, η, Δτ, Δ)
    set_bcs!(model.boundary_conditions.velocity)
    # rheology
    update_η!(backend, 256, (nx, ny, nz))(interior(η), η_rh, η_rel, model.grid, model.fields)
    extrapolate!(data(η))

    async || synchronize(backend)
    return
end

function advance_timestep!(model::IsothermalFullStokesModel, t, Δt)
    # TODO
    
    return
end

end