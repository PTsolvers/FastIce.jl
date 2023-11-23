module Isothermal

export BoundaryCondition, Traction, Velocity, Slip
export IsothermalFullStokesModel, advance_iteration!, advance_timestep!, evaluate_error!

using FastIce.Architectures
using FastIce.Physics
using FastIce.Grids
using FastIce.Fields
using FastIce.BoundaryConditions
using FastIce.Utils
using FastIce.KernelLaunch
using FastIce.Distributed

include("kernels.jl")

include("boundary_conditions.jl")

function default_physics(::Type{T}) where {T}
    return (equation_of_state=default(IncompressibleIceEOS{T}),
            thermal_properties=default(IceThermalProperties{T}),
            rheology=default(GlensLawRheology{Int64}))
end

struct IsothermalFullStokesModel{Arch,Grid,BC,HB,OW,Physics,Gravity,IterParams,Fields}
    arch::Arch
    grid::Grid
    boundary_conditions::BC
    hide_boundaries::HB
    outer_width::OW
    physics::Physics
    gravity::Gravity
    iter_params::IterParams
    fields::Fields
end

function make_fields_mechanics(backend, grid::CartesianGrid{2})
    return (Pr=Field(backend, grid, Center(); halo=1),
            # deviatoric stress
            τ=(xx=Field(backend, grid, Center(); halo=1),
               yy=Field(backend, grid, Center(); halo=1),
               xy=Field(backend, grid, (Vertex(), Vertex()))),
            # velocity
            V=(x=Field(backend, grid, (Vertex(), Center()); halo=1),
               y=Field(backend, grid, (Center(), Vertex()); halo=1)),
            # residual
            r_Pr=Field(backend, grid, Center(); halo=1),
            r_V=(x=Field(backend, grid, (Vertex(), Center()); halo=1),
                 y=Field(backend, grid, (Center(), Vertex()); halo=1)))
end

function make_fields_mechanics(backend, grid::CartesianGrid{3})
    return (Pr=Field(backend, grid, Center(); halo=1),
            # deviatoric stress
            τ=(xx=Field(backend, grid, Center(); halo=1),
               yy=Field(backend, grid, Center(); halo=1),
               zz=Field(backend, grid, Center(); halo=1),
               xy=Field(backend, grid, (Vertex(), Vertex(), Center()); halo=0),
               xz=Field(backend, grid, (Vertex(), Center(), Vertex()); halo=0),
               yz=Field(backend, grid, (Center(), Vertex(), Vertex()); halo=0)),
            # velocity
            V=(x=Field(backend, grid, (Vertex(), Center(), Center()); halo=1),
               y=Field(backend, grid, (Center(), Vertex(), Center()); halo=1),
               z=Field(backend, grid, (Center(), Center(), Vertex()); halo=1)),
            # residual
            r_Pr=Field(backend, grid, Center(); halo=0),
            r_V=(x=Field(backend, grid, (Vertex(), Center(), Center()); halo=1),
                 y=Field(backend, grid, (Center(), Vertex(), Center()); halo=1),
                 z=Field(backend, grid, (Center(), Center(), Vertex()); halo=1)))
end

function IsothermalFullStokesModel(;
                                   arch,
                                   grid,
                                   boundary_conditions,
                                   iter_params,
                                   gravity,
                                   outer_width=(2, 2, 2),
                                   physics=nothing,
                                   fields=nothing,
                                   other_fields=nothing)
    if isnothing(fields)
        mechanic_fields = make_fields_mechanics(backend(arch), grid)
        rheology_fields = (η=Field(backend(arch), grid, Center(); halo=1),)
        fields = merge(mechanic_fields, rheology_fields)
    end

    if !isnothing(other_fields)
        fields = merge(fields, other_fields)
    end

    if isnothing(physics)
        physics = default_physics(eltype(grid))
    end

    boundary_conditions = make_field_boundary_conditions(arch, grid, fields, boundary_conditions)
    hide_boundaries = HideBoundaries{ndims(grid)}(arch)

    return IsothermalFullStokesModel(arch, grid, boundary_conditions, hide_boundaries, outer_width, physics, gravity, iter_params, fields)
end

fields(model::IsothermalFullStokesModel) = model.fields
grid(model::IsothermalFullStokesModel) = model.grid

function advance_iteration!(model::IsothermalFullStokesModel, t, Δt; async=true)
    (; Pr, τ, V, η) = model.fields
    (; η_rel, Δτ)   = model.iter_params
    η_rh            = model.physics.rheology
    ρg              = model.gravity
    hide_boundaries = model.hide_boundaries
    outer_width     = model.outer_width

    Δ = NamedTuple{(:x, :y, :z)}(spacing(model.grid))

    launch!(model.arch, model.grid, update_σ! => (Pr, τ, V, η, Δτ, Δ);
            location=Center(), expand=1, boundary_conditions=model.boundary_conditions.stress, hide_boundaries, outer_width)

    launch!(model.arch, model.grid, update_V! => (V, Pr, τ, η, ρg, Δτ, model.grid, Δ);
            location=Vertex(), boundary_conditions=model.boundary_conditions.velocity, hide_boundaries, outer_width)

    # rheology
    launch!(model.arch, model.grid, update_η! => (η, η_rh, η_rel, model.grid, model.fields);
            location=Center(), boundary_conditions=model.boundary_conditions.rheology, hide_boundaries, outer_width)

    async || synchronize(backend(model.arch))
    return
end

function evaluate_error!(model::IsothermalFullStokesModel; async=true)
    (; Pr, τ, V, r_Pr, r_V) = model.fields
    ρg              = model.gravity
    hide_boundaries = model.hide_boundaries
    outer_width     = model.outer_width

    Δ = NamedTuple{(:x, :y, :z)}(spacing(model.grid))

    launch!(model.arch, model.grid, compute_residuals! => (r_V, r_Pr, Pr, τ, V, ρg, model.grid, Δ);
            location=Vertex(), boundary_conditions=model.boundary_conditions.residuals_vel, hide_boundaries, outer_width)

    async || synchronize(backend(model.arch))
    return
end

function advance_timestep!(model::IsothermalFullStokesModel, t, Δt)
    # TODO

    return
end

end
