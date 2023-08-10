using FastIce
using FastIce.Grids
using FastIce.Fields
using FastIce.Utils
using FastIce.BoundaryConditions
using FastIce.Models.FullStokes.Isothermal

using KernelAbstractions

grid = CartesianGrid(
    origin = (-0.5, -0.5, 0.0),
    extent = ( 1.0,  1.0, 1.0),
    size   = ( 10,  10, 10);
)

boundary_conditions = (
    west  = BoundaryCondition{Velocity}(0.0, 0.0, 0.0),
    east  = BoundaryCondition{Velocity}(0.0, 0.0, 0.0),
    south = BoundaryCondition{Velocity}(0.0, 0.0, 0.0),
    north = BoundaryCondition{Velocity}(0.0, 0.0, 0.0),
    bot   = BoundaryCondition{Velocity}(0.0, 0.0, 0.0),
    top   = BoundaryCondition{Velocity}(0.0, 1.0, 0.0),
)

iter_params = (
    η_rel = 1e-1,
    Δτ = ( Pr = 1.0, τ = (xx = 1.0, yy = 1.0, zz = 1.0, xy = 1.0, xz = 1.0, yz = 1.0), V = (x = 1.0, y = 1.0, z = 1.0)),
)

model = IsothermalFullStokesModel(;
    backend = CPU(),
    grid,
    boundary_conditions,
    iter_params,
)

set!(model.fields.Pr, 0.0)
foreach(x -> set!(x, 0.0), model.fields.τ)
foreach(x -> set!(x, 0.0), model.fields.V)

init_incl(x, y, z, x0, y0, z0, r, ηi, ηm) = ifelse((x-x0)^2 + (y-y0)^2 + (z-z0)^2 < r^2, ηi, ηm)
set!(model.fields.η, grid, init_incl; continuous = true, parameters = (x0 = 0.0, y0 = 0.0, z0 = 0.5, r = 0.2, ηi = 1e-6, ηm = 1.0))
extrapolate!(data(model.fields.η))

advance_iteration!(model, 0.0, 1.0; async = false)