module Heat

export BoundaryCondition, Flux, Value
export HeatDiffusionModel, advance_iteration!, advance_timestep!

using FastIce.Grids
using FastIce.Fields
using FastIce.BoundaryConditions
using FastIce.Utils

include("kernels.jl")

struct HeatDiffusionModel{Backend,Grid,BC,Physics,IterParams,Fields}
    backend::Backend
    grid::Grid
    boundary_conditions::BC
    physics::Physics
    iter_params::IterParams
    fields::Fields
end

function make_fields_diffusion(backend, grid)
    return (
        q = (
            x = Field(backend, grid, (Vertex(), Center(), Center()); halo=1),
            y = Field(backend, grid, (Center(), Vertex(), Center()); halo=1),
            z = Field(backend, grid, (Center(), Center(), Vertex()); halo=1),
        ),
    )
end

function HeatDiffusionModel(; backend, grid, boundary_conditions, physics=nothing, iter_params, fields=nothing, init_fields=nothing)
    if isnothing(fields)
        diffusion_fields = make_fields_diffusion(backend, grid)
        fields = diffusion_fields
    end

    if !isnothing(init_fields)
        fields = merge(fields, init_fields)
    end

    boundary_conditions = create_field_boundary_conditions(boundary_conditions)

    return HeatDiffusionModel(backend, grid, boundary_conditions, physics, iter_params, fields)
end

fields(model::HeatDiffusionModel) = model.fields
grid(model::HeatDiffusionModel) = model.grid

struct Value end
struct Flux end

struct BoundaryCondition{Kind, Val}
    val::Val
end

BoundaryCondition{Kind}(val::Val) where {Kind, Val} = BoundaryCondition{Kind, Val}(val)

function extract_x_bcs(bc::BoundaryCondition{Value})
    return (T = DirichletBC{HalfCell}(bc.val),), NamedTuple()
end

function extract_y_bcs(bc::BoundaryCondition{Value})
    return (T = DirichletBC{HalfCell}(bc.val),), NamedTuple()
end

function extract_z_bcs(bc::BoundaryCondition{Value})
    return (T = DirichletBC{HalfCell}(bc.val),), NamedTuple()
end

function extract_x_bcs(bc::BoundaryCondition{Flux})
    return NamedTuple(), (qx = DirichletBC{FullCell}(bc.val),)
end

function extract_y_bcs(bc::BoundaryCondition{Flux})
    return NamedTuple(), (qy = DirichletBC{FullCell}(bc.val),)
end

function extract_z_bcs(bc::BoundaryCondition{Flux})
    return NamedTuple(), (qz = DirichletBC{FullCell}(bc.val),)
end

@inline no_bcs(names) = NamedTuple(f => NoBC() for f in names)

unique_names(a, b) = Tuple(unique(tuple(a..., b...)))

function create_field_boundary_conditions(f, left, right)
    left_value , left_flux  = f(left)
    right_value, right_flux = f(right)

    value_names = unique_names(keys(left_value), keys(right_value))
    flux_names  = unique_names(keys(left_flux) , keys(right_flux))

    default_value = no_bcs(value_names)
    default_flux  = no_bcs(flux_names)

    left_value  = merge(default_value, left_value)
    right_value = merge(default_value, right_value)

    left_flux  = merge(default_flux, left_flux)
    right_flux = merge(default_flux, right_flux)

    return value_names, left_value, right_value, flux_names, left_flux, right_flux
end

function create_field_boundary_conditions(bcs)
    value_names_x, west_value_bcs , east_value_bcs , flux_names_x, west_flux_bcs , east_flux_bcs  = create_field_boundary_conditions(extract_x_bcs, bcs.west , bcs.east)
    value_names_y, south_value_bcs, north_value_bcs, flux_names_y, south_flux_bcs, north_flux_bcs = create_field_boundary_conditions(extract_y_bcs, bcs.south, bcs.north)
    value_names_z, bot_value_bcs  , top_value_bcs  , flux_names_z, bot_flux_bcs  , top_flux_bcs   = create_field_boundary_conditions(extract_z_bcs, bcs.bot  , bcs.top)
    return (
        value = (
            x = (
                names = value_names_x,
                left  = west_value_bcs,
                right = east_value_bcs,
            ),
            y = (
                names = value_names_y,
                left  = south_value_bcs,
                right = north_value_bcs,
            ),
            z = (
                names = value_names_z,
                left  = bot_value_bcs,
                right = top_value_bcs,
            )
        ),
        flux = (
            x = (
                names = flux_names_x,
                left  = west_flux_bcs,
                right = east_flux_bcs,
            ),
            y = (
                names = flux_names_y,
                left  = south_flux_bcs,
                right = north_flux_bcs,
            ),
            z = (
                names = flux_names_z,
                left  = bot_flux_bcs,
                right = top_flux_bcs,
            )
        )
    )
end

function apply_bcs!(backend, grid, fields, bcs)
    field_map = (T = fields.T,
        qx = fields.q.x ,  qy = fields.q.y ,  qz = fields.q.z)

    function apply_bcs_dim!(f, dim)
        fields    = Tuple( field_map[f] for f in dim.names )
        left_bcs  = values(dim.left)
        right_bcs = values(dim.right)
        f(grid, fields, left_bcs, right_bcs)
    end

    apply_bcs_dim!(discrete_bcs_x!(backend, 256, (size(grid, 2)+1, size(grid, 3)+1)), bcs.x)
    apply_bcs_dim!(discrete_bcs_y!(backend, 256, (size(grid, 1)+1, size(grid, 3)+1)), bcs.y)
    apply_bcs_dim!(discrete_bcs_z!(backend, 256, (size(grid, 1)+1, size(grid, 2)+1)), bcs.z)

    return
end

function advance_iteration!(model::HeatDiffusionModel, t, Δt; async = true)
    (; T, T_o, q) = model.fields
    (; Δτ) = model.iter_params
    λ_ρCp = model.physics.properties
    Δ = NamedTuple{(:x, :y, :z)}(spacing(model.grid))
    nx, ny, nz = size(model.grid)
    backend = model.backend

    set_bcs!(bcs) = apply_bcs!(model.backend, model.grid, model.fields, bcs)

    # flux
    update_q!(backend, 256, (nx+1, ny+1, nz+1))(q, T, λ_ρCp, Δτ, Δ)
    set_bcs!(model.boundary_conditions.flux)
    # mass balance
    update_T!(backend, 256, (nx, ny, nz))(T, T_o, q, Δt, Δτ, Δ)
    set_bcs!(model.boundary_conditions.value)

    async || synchronize(backend)
    return
end

function advance_timestep!(model::HeatDiffusionModel, t, Δt)
    # TODO
    
    return
end

end