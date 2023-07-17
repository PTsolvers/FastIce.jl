module Isothermal

using FastIce.Physics

const DEFAULT_PHYSICS = (
    equation_of_state  = default(IncompressibleIceEOS),
    thermal_properties = default(IceThermalProperties),
    rheology           = default(GlensLawRheology)
)

struct IsothermalFullStokesModel{Grid,BC,Physics,Numerics,Fields} <: AbstractModel
    grid::Grid
    boundary_conditions::BC
    physics::Physics
    numerics::Numerics
    fields::Fields
end

function IsothermalFullStokesModel(; grid, boundary_conditions, phyiscs = DEFAULT_PHYSICS, numerics, fields = nothing)
    if isnothing(fields)
        fields = make_fields_mechanics(grid, boundary_conditions)
    end

    return IsothermalFullStokesModel(grid, boundary_conditions, phyiscs, numerics, fields)
end

fields(model::IsothermalFullStokesModel) = model.fields
grid(model::IsothermalFullStokesModel)   = model.grid

end