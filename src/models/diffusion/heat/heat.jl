module Heat

export BoundaryCondition, Flux
export HeatDiffusionModel, advance_iteration!, advance_timestep!

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
    )
end

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
        T = Field(backend, grid, Center(); halo=1),
        q = (
            x = Field(backend, grid, (Vertex(), Center(), Center()); halo=1),
            y = Field(backend, grid, (Center(), Vertex(), Center()); halo=1),
            z = Field(backend, grid, (Center(), Center(), Vertex()); halo=1),
        )
    )
end

function make_fields_params(backend, grid)
    return (λ = Field(backend, grid, Center(); halo=1),)
end

function HeatDiffusionModel(; backend, grid, boundary_conditions, physics=nothing, iter_params, fields=nothing, other_fields=nothing)
    if isnothing(fields)
        diffusion_fields = make_fields_diffusion(backend, grid)
        params_fields = make_fields_params(backend, grid)
        fields = merge(diffusion_fields, params_fields)
    end

    if !isnothing(other_fields)
        fields = merge(fields, other_fields)
    end

    if isnothing(physics)
        physics = default_physics(eltype(grid))
    end

    boundary_conditions = create_field_boundary_conditions(boundary_conditions)

    return HeatDiffusionModel(backend, grid, boundary_conditions, physics, iter_params, fields)
end

fields(model::HeatDiffusionModel) = model.fields
grid(model::HeatDiffusionModel) = model.grid

struct Flux end

struct BoundaryCondition{Kind, Fx, Fy, Fz}
    x::Fx
    y::Fy
    z::Fz
end

BoundaryCondition{Kind}(x::Fx, y::Fy, z::Fz) where {Kind, Fx, Fy, Fz} = BoundaryCondition{Kind, Fx, Fy, Fz}(x, y, z)

function extract_x_bcs(bc::BoundaryCondition{Flux})
    return NamedTuple(),
          (qx = DirichletBC{FullCell}(bc.x),
           qy = DirichletBC{HalfCell}(bc.y),
           qz = DirichletBC{HalfCell}(bc.z),)
end

function extract_y_bcs(bc::BoundaryCondition{Flux})
    return NamedTuple(),
          (qx = DirichletBC{HalfCell}(bc.x),
           qy = DirichletBC{FullCell}(bc.y),
           qz = DirichletBC{HalfCell}(bc.z),)
end

function extract_z_bcs(bc::BoundaryCondition{Flux})
    return NamedTuple(),
           (qx = DirichletBC{HalfCell}(bc.x),
            qy = DirichletBC{HalfCell}(bc.y),
            qz = DirichletBC{FullCell}(bc.z),)
end

@inline no_bcs(names) = NamedTuple(f => NoBC() for f in names)

unique_names(a, b) = Tuple(unique(tuple(a..., b...)))

function create_field_boundary_conditions(f, left, right)
    left_flux  = f(left)
    right_flux = f(right)

    flux_names = unique_names(keys(left_flux), keys(right_flux))

    default_flux = no_bcs(flux_names)

    left_flux  = merge(default_flux, left_flux)
    right_flux = merge(default_flux, right_flux)

    return flux_names, left_flux, right_flux
end

function create_field_boundary_conditions(bcs)
    flux_names_x, west_flux_bcs , east_flux_bcs  = create_field_boundary_conditions(extract_x_bcs, bcs.west, bcs.east)
    flux_names_y, south_flux_bcs, north_flux_bcs = create_field_boundary_conditions(extract_y_bcs, bcs.south, bcs.north)
    flux_names_z, bot_flux_bcs  , top_flux_bcs   = create_field_boundary_conditions(extract_z_bcs, bcs.bot  , bcs.top)
    return (
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
    field_map = (qx = fields.q.x ,  qy = fields.q.y ,  qz = fields.q.z)

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
    (; T, q, λ) = model.fields
    (; η_rel, Δτ) = model.iter_params
    λ_tp = model.physics.thermal_properties # DEBUG
    ρ_eos, λ_eos = model.physics.equation_of_state # DEBUG
    Δ = NamedTuple{(:x, :y, :z)}(spacing(model.grid))
    nx, ny, nz = size(model.grid)
    backend = model.backend

    # set_bcs!(bcs) = apply_bcs!(model.backend, model.grid, model.fields, bcs)

    # flux
    # update_q!(backend, 256, (nx+1, ny+1, nz+1))(q, T, λ, Δτ, Δ)
    # set_bcs!(model.boundary_conditions.flux)
    # mass balance
    # update_T!(backend, 256, (nx, ny, nz))(T, q, ρcp, Δτ, Δ)

    async || synchronize(backend)
    return
end

function advance_timestep!(model::HeatDiffusionModel, t, Δt)
    # TODO
    
    return
end

end