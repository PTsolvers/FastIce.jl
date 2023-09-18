module Isothermal

export BoundaryCondition, Traction, Velocity, Slip
export IsothermalFullStokesModel, advance_iteration!, advance_timestep!

using FastIce.Physics
using FastIce.Grids
using FastIce.Fields
using FastIce.BoundaryConditions
using FastIce.Utils

include("kernels.jl")

include("boundary_conditions.jl")

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
        Pr = Field(backend, grid, Center(); halo=1),
        τ  = (
            xx = Field(backend, grid, Center(); halo=1),
            yy = Field(backend, grid, Center(); halo=1),
            zz = Field(backend, grid, Center(); halo=1),
            xy = Field(backend, grid, (Vertex(), Vertex(), Center()); halo=0),
            xz = Field(backend, grid, (Vertex(), Center(), Vertex()); halo=0),
            yz = Field(backend, grid, (Center(), Vertex(), Vertex()); halo=0),
        ),
        V = (
            x = Field(backend, grid, (Vertex(), Center(), Center()); halo=1),
            y = Field(backend, grid, (Center(), Vertex(), Center()); halo=1),
            z = Field(backend, grid, (Center(), Center(), Vertex()); halo=1),
        )
    )
end



function IsothermalFullStokesModel(; backend, grid, boundary_conditions, physics=nothing, iter_params, fields=nothing, other_fields=nothing)
    if isnothing(fields)
        mechanic_fields = make_fields_mechanics(backend, grid)
        rheology_fields = (η = Field(backend, grid, Center(); halo=1),)
        fields = merge(mechanic_fields, rheology_fields)
    end

    if !isnothing(other_fields)
        fields = merge(fields, other_fields)
    end

    if isnothing(physics)
        physics = default_physics(eltype(grid))
    end

    boundary_conditions = make_field_boundary_conditions(boundary_conditions)

    return IsothermalFullStokesModel(backend, grid, boundary_conditions, physics, iter_params, fields)
end

fields(model::IsothermalFullStokesModel) = model.fields
grid(model::IsothermalFullStokesModel) = model.grid

function advance_iteration!(model::IsothermalFullStokesModel, t, Δt; async = true)
    (; Pr, τ, V, η) = model.fields
    (; η_rel, Δτ) = model.iter_params
    η_rh = model.physics.rheology
    Δ = NamedTuple{(:x, :y, :z)}(spacing(model.grid))
    nx, ny, nz = size(model.grid)
    backend = model.backend

    set_bcs!(bcs) = _apply_bcs!(model.backend, model.grid, model.fields, bcs)

    # stress

    # launch!(arch, grid, update_σ!, Pr, τ, V, η, Δτ, Δ)

    update_σ!(backend, 256, (nx+1, ny+1, nz+1))(Pr, τ, V, η, Δτ, Δ)
    set_bcs!(model.boundary_conditions.stress)
    # velocity

    # launch!(arch, grid, (update_res_V! => (rV, V, Pr, τ, η, Δτ, Δ), update_V! => (V, rV, dt)); exchangers = exchangers.velocity, boundary_conditions = boundary_conditions.velocity)
    update_V!(backend, 256, (nx+1, ny+1, nz+1))(V, Pr, τ, η, Δτ, Δ)
    set_bcs!(model.boundary_conditions.velocity)
    # rheology
    update_η!(backend, 256, (nx, ny, nz))(η, η_rh, η_rel, model.grid, model.fields)
    extrapolate!(η)

    async || synchronize(backend)
    return
end

function advance_timestep!(model::IsothermalFullStokesModel, t, Δt)
    # TODO
    
    return
end

end