module Isothermal

export BoundaryCondition, Traction, Velocity, Slip
export IsothermalFullStokesModel, advance_iteration!, advance_timestep!, compute_residuals!

using FastIce.Architectures
using FastIce.Physics
using FastIce.Grids
using FastIce.Fields
using FastIce.BoundaryConditions
using FastIce.Utils
using FastIce.KernelLaunch
using FastIce.Distributed

using FastIce.GridOperators
using KernelAbstractions

include("kernels_common.jl")
include("kernels_2d.jl")
include("kernels_3d.jl")

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
            r_Pr=Field(backend, grid, Center()),
            r_V=(x=Field(backend, grid, (Vertex(), Center())),
                 y=Field(backend, grid, (Center(), Vertex()))))
end

function make_fields_mechanics(backend, grid::CartesianGrid{3})
    return (Pr=Field(backend, grid, Center(); halo=1),
            # deviatoric stress
            τ=(xx=Field(backend, grid, Center(); halo=1),
               yy=Field(backend, grid, Center(); halo=1),
               zz=Field(backend, grid, Center(); halo=1),
               xy=Field(backend, grid, (Vertex(), Vertex(), Center())),
               xz=Field(backend, grid, (Vertex(), Center(), Vertex())),
               yz=Field(backend, grid, (Center(), Vertex(), Vertex()))),
            # velocity
            V=(x=Field(backend, grid, (Vertex(), Center(), Center()); halo=1),
               y=Field(backend, grid, (Center(), Vertex(), Center()); halo=1),
               z=Field(backend, grid, (Center(), Center(), Vertex()); halo=1)),
            # residual
            r_Pr=Field(backend, grid, Center()),
            r_V=(x=Field(backend, grid, (Vertex(), Center(), Center())),
                 y=Field(backend, grid, (Center(), Vertex(), Center())),
                 z=Field(backend, grid, (Center(), Center(), Vertex()))))
end

function IsothermalFullStokesModel(;
                                   arch,
                                   grid,
                                   boundary_conditions,
                                   iter_params,
                                   gravity,
                                   outer_width=nothing,
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

    if isnothing(outer_width)
        outer_width = ntuple(_ -> 2, Val(ndims(grid)))
    end

    boundary_conditions = make_field_boundary_conditions(arch, grid, fields, boundary_conditions)
    hide_boundaries = HideBoundaries{ndims(grid)}(arch)

    return IsothermalFullStokesModel(arch, grid, boundary_conditions, hide_boundaries, outer_width, physics, gravity, iter_params, fields)
end

fields(model::IsothermalFullStokesModel) = model.fields
grid(model::IsothermalFullStokesModel) = model.grid

function advance_iteration!(model::IsothermalFullStokesModel, t, Δt)
    (; Pr, τ, V, η) = model.fields
    (; η_rel, Δτ)   = model.iter_params
    η_rh            = model.physics.rheology
    ρg              = model.gravity
    bc              = model.boundary_conditions
    hide_boundaries = model.hide_boundaries
    outer_width     = model.outer_width
    grid            = model.grid

    Δ = spacing(model.grid)

    launch!(model.arch, grid, update_σ! => (Pr, τ, V, η, Δτ, Δ, grid);
            location=Center(), expand=1, boundary_conditions=bc.stress, hide_boundaries, outer_width, async=false)

    launch!(model.arch, grid, update_V! => (V, Pr, τ, η, ρg, Δτ, Δ, grid);
            location=Vertex(), boundary_conditions=bc.velocity, hide_boundaries, outer_width, async=false)

    # rheology
    launch!(model.arch, grid, update_η! => (η, η_rh, η_rel, model.fields, grid);
            location=Center(), boundary_conditions=bc.rheology, hide_boundaries, outer_width, async=false)
    return
end

function compute_residuals!(model::IsothermalFullStokesModel)
    (; Pr, τ, V, r_Pr, r_V) = model.fields
    ρg = model.gravity
    boundary_conditions = model.boundary_conditions.residual

    Δ = spacing(model.grid)

    launch!(model.arch, model.grid, compute_residuals! => (r_V, r_Pr, Pr, τ, V, ρg, Δ, model.grid);
            location=Vertex(), boundary_conditions, async=false)
    return
end

function advance_timestep!(model::IsothermalFullStokesModel, t, Δt)
    # TODO

    return
end

end
