module Isothermal

using FastIce.Physics

include("kernels.jl")

const DEFAULT_PHYSICS = (
    equation_of_state=default(IncompressibleIceEOS),
    thermal_properties=default(IceThermalProperties),
    rheology=default(GlensLawRheology)
)

struct IsothermalFullStokesModel{Backend,Grid,BC,Physics,Numerics,Fields} <: AbstractModel
    backend::Backend
    grid::Grid
    boundary_conditions::BC
    physics::Physics
    numerics::Numerics
    fields::Fields
end

function IsothermalFullStokesModel(; backend, grid, boundary_conditions, phyiscs=DEFAULT_PHYSICS, numerics, fields=nothing)
    if isnothing(fields)
        fields = make_fields_mechanics(backend, grid, boundary_conditions)
    end

    return IsothermalFullStokesModel(backend, grid, boundary_conditions, phyiscs, numerics, fields)
end

fields(model::IsothermalFullStokesModel) = model.fields
grid(model::IsothermalFullStokesModel) = model.grid

# In x direction:
# Pr[1] = - Pr[2] + 2*px
# τ.xx[1] = -τ.xx[2]
# τ.yz[1] = py
# τ.xz[1] = pz
# struct PrescribedTractionBC{Tx,Ty,Tz}
#     px::Tx
#     py::Ty
#     pz::Tz
# end

# function extract_x_bcs!(bc::PrescribedTractionBC)
#     (V.x, NoBC())
#     (V.y, NoBC())
#     (V.z, NoBC())

#     (Pr, DirichletBC{Shift}(bc.px))

#     (τ.xx, DirichletBC{NoShift}(0.0))
#     (τ.yy, NoBC())
#     (τ.zz, NoBC())

#     (τ.xy, DirichletBC{NoShift}(bc.py))
#     (τ.xz, DirichletBC{NoShift}(bc.pz))
#     (τ.yz, NoBC())
# end

# function extract_y_bcs!(bc::PrescribedTractionBC)
#     (V.x, NoBC())
#     (V.y, NoBC())
#     (V.z, NoBC())

#     (Pr, DirichletBC{Shift}(bc.py))

#     (τ.xx, NoBC())
#     (τ.yy, DirichletBC{NoShift}(0.0))
#     (τ.zz, NoBC())

#     (τ.xy, DirichletBC{NoShift}(bc.px))
#     (τ.xz, NoBC())
#     (τ.yz, DirichletBC{NoShift}(bc.pz))
# end

# function extract_z_bcs!(bc::PrescribedTractionBC)
#     (V.x, NoBC())
#     (V.y, NoBC())
#     (V.z, NoBC())

#     (Pr, DirichletBC{Shift}(bc.pz))

#     (τ.xx, NoBC())
#     (τ.yy, NoBC())
#     (τ.zz, DirichletBC{NoShift}(0.0))

#     (τ.xy, NoBC())
#     (τ.xz, DirichletBC{NoShift}(bc.px))
#     (τ.yz, DirichletBC{NoShift}(bc.py))
# end

# # In x direction:
# # V.x[1] = vn
# # τyz[1] = px
# # τxz[1] = py
# struct FreeSlipBC{Tn,Tx,Ty}
#     vn::Tn
#     px::Tx
#     py::Ty
# end

# function extract_x_bcs!(bc::FreeSlipBC)
#     (V.x, DirichletBC{NoShift}(bc.vn))
#     (V.y, NoBC())
#     (V.z, NoBC())

#     (Pr, NoBC())

#     (τ.xx, NoBC())
#     (τ.yy, NoBC())
#     (τ.zz, NoBC())

#     (τ.xy, DirichletBC{Shift}(bc.px))
#     (τ.xz, DirichletBC{Shift}(bc.py))
#     (τ.yz, NoBC)
# end

# function extract_y_bcs!(bc::FreeSlipBC)
#     (V.x, NoBC())
#     (V.y, DirichletBC{NoShift}(bc.vn))
#     (V.z, NoBC())

#     (Pr, NoBC())

#     (τ.xx, NoBC())
#     (τ.yy, NoBC())
#     (τ.zz, NoBC())

#     (τ.xy, DirichletBC{NoShift}(bc.px))
#     (τ.xz, NoBC())
#     (τ.yz, DirichletBC{NoShift}(bc.py))
# end

# function extract_y_bcs!(bc::FreeSlipBC)
#     (V.x, NoBC())
#     (V.y, NoBC())
#     (V.z, DirichletBC{NoShift}(bc.vn))

#     (Pr, NoBC())

#     (τ.xx, NoBC())
#     (τ.yy, NoBC())
#     (τ.zz, NoBC())

#     (τ.xy, NoBC())
#     (τ.xz, DirichletBC{NoShift}(bc.px))
#     (τ.yz, DirichletBC{NoShift}(bc.py))
# end

# # In x direction
# # Vx[1] = vx
# # Vy[1] = -Vy[2] + 2*vy
# # Vx[1] = -Vz[2] + 2*vz
# struct PrescribedVelocityBC{Tx,Ty,Tz}
#     vx::Tx
#     vy::Ty
#     vz::Tz
# end

# function extract_x_bcs!(bc::PrescribedVelocityBC)
#     (V.x, DirichletBC{NoShift}(bc.vx))
#     (V.y, DirichletBC{Shift}(bc.vy))
#     (V.z, DirichletBC{Shift}(bc.vz))

#     (Pr, NoBC())

#     (τ.xx, NoBC())
#     (τ.yy, NoBC())
#     (τ.zz, NoBC())
    
#     (τ.xy, NoBC())
#     (τ.xz, NoBC())
#     (τ.yz, NoBC())
# end

# function extract_y_bcs!(bc::PrescribedVelocityBC)
#     (V.x, DirichletBC{Shift}(bc.vx))
#     (V.y, DirichletBC{NoShift}(bc.vy))
#     (V.z, DirichletBC{Shift}(bc.vz))

#     (Pr, NoBC())

#     (τ.xx, NoBC())
#     (τ.yy, NoBC())
#     (τ.zz, NoBC())
    
#     (τ.xy, NoBC())
#     (τ.xz, NoBC())
#     (τ.yz, NoBC())
# end

# function extract_z_bcs!(bc::PrescribedVelocityBC)
#     (V.x, DirichletBC{Shift}(bc.vx))
#     (V.y, DirichletBC{Shift}(bc.vy))
#     (V.z, DirichletBC{NoShift}(bc.vz))

#     (Pr, NoBC())

#     (τ.xx, NoBC())
#     (τ.yy, NoBC())
#     (τ.zz, NoBC())
    
#     (τ.xy, NoBC())
#     (τ.xz, NoBC())
#     (τ.yz, NoBC())
# end

# function set_stress_bcs!(model::IsothermalFullStokesModel)
#     nx, ny, nz = size(model.grid)
#     discrete_bcs_x!(model.backend, 256, (ny, nz))(f, grid, west_ix, east_ix, west_bc, east_bc)
#     discrete_bcs_y!(model.backend, 256, (nx, nz))(f, grid, west_ix, east_ix, west_bc, east_bc)
#     discrete_bcs_z!(model.backend, 256, (nx, ny))(f, grid, west_ix, east_ix, west_bc, east_bc)
#     return
# end

# function set_velocity_bcs!(model::IsothermalFullStokesModel)
#     # TODO
# end

function advance_iteration!(model::IsothermalFullStokesModel, t, Δt; async = true)
    (; Pr, τ, V, η) = model.fields
    (; η_rh) = model.physics.rheology
    (; η_rel, Δτ) = model.iter_params
    Δ = spacing(model.grid)
    nx, ny, nz = size(model.grid)
    backend = model.backend

    # stress
    update_σ!(backend, 256, (nx + 2, ny + 2, nz + 2))(Pr, τ, V, η, Δτ, Δ)
    set_stress_bcs!(Pr, τ, V, Δ, model.boundary_conditions)
    # velocity
    update_V!(backend, 256, (nx + 1, ny + 1, nz + 1))(V, Pr, τ, η, Δτ, Δ)
    set_velocity_bcs!(V, Pr, τ, Δ, model.boundary_conditions)
    # rheology
    update_η!(backend, 256, (nx, ny, nz))(η, τ, η_rh, η_rel)
    extrapolate!(η)

    async || synchronize(backend)
    return
end

function advance_timestep!(model::IsothermalFullStokesModel, t, Δt)
    # TODO
    
    return
end
