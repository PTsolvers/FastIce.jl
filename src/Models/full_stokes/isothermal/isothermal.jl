module Isothermal

export BoundaryCondition, Traction, Velocity, Slip
export IsothermalFullStokesModel, advance_iteration!, advance_timestep!

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

struct IsothermalFullStokesModel{Arch,Grid,BC,Physics,IterParams,Fields}
    arch::Arch
    grid::Grid
    boundary_conditions::BC
    physics::Physics
    iter_params::IterParams
    fields::Fields
end

function make_fields_mechanics(backend, grid)
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
               z=Field(backend, grid, (Center(), Center(), Vertex()); halo=1)))
end

function IsothermalFullStokesModel(; arch, grid, boundary_conditions, physics=nothing, iter_params, fields=nothing, other_fields=nothing)
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

    boundary_conditions = make_field_boundary_conditions(grid, fields, boundary_conditions)

    if arch isa Architecture{Distributed.DistributedMPI}
        stress = override_boundary_conditions(arch, boundary_conditions.stress)
        velocity = override_boundary_conditions(arch, boundary_conditions.velocity; exchange=true)
        boundary_conditions = (; stress, velocity)
    end

    return IsothermalFullStokesModel(arch, grid, boundary_conditions, physics, iter_params, fields)
end

fields(model::IsothermalFullStokesModel) = model.fields
grid(model::IsothermalFullStokesModel) = model.grid

function advance_iteration!(model::IsothermalFullStokesModel, t, Δt; async=true)
    (; Pr, τ, V, η) = model.fields
    (; η_rel, Δτ) = model.iter_params
    η_rh = model.physics.rheology
    Δ = NamedTuple{(:x, :y, :z)}(spacing(model.grid))

    launch!(model.arch, model.grid, update_σ! => (Pr, τ, V, η, Δτ, Δ);
            location=Vertex(), expand=1, boundary_conditions=model.boundary_conditions.stress)

    launch!(model.arch, model.grid, update_V! => (V, Pr, τ, η, Δτ, Δ);
            location=Vertex(), boundary_conditions=model.boundary_conditions.velocity)

    # rheology
    launch!(model.arch, model.grid, update_η! => (η, η_rh, η_rel, model.grid, model.fields); location=Center())
    extrapolate!(η)

    async || synchronize(backend(model.arch))
    return
end

function advance_timestep!(model::IsothermalFullStokesModel, t, Δt)
    # TODO

    return
end

end
