## TODOs
# 1. How to parametrise boundary conditions for energy and ice flow for the free surface case
# 2. How to efficiently parametrise multiphysics

using FastIce
using FastIce.Geometry.SDF
using FastIce.Geometry.LevelSet
using FastIce.Data.DEM
using FastIce.Thermodynamics.EOS
# using FastIce.Models.Thermal
# using FastIce.Models.FullStokes.Isothermal
# using FastIce.Models.FullStokes.IsothermalPlasticity
# using FastIce.Models.FullStokes.Thermomechanical
using FastIce.Models.FullStokes.ThermomechanicalWithTopo

## Supported topologies
# 1. Bounded
# 2. Periodic
grid = CartesianGrid(
    origin = (-0.5, -0.5, 0.0),
    extent = ( 1.0,  1.0, 1.0),
    size   = ( 100,  100, 100);
)

sphere_ice = SDF.Sphere(radius = 1  , origin = (0.0,  0.0 ))
# sphere_bed = SDF.Sphere(radius = 100, origin = (0.0, -90.0))
sphere_bed = LevelSet(DEM("data/greenland.h5", "bed"))

not_air = !(sphere_bed ∪ sphere_ice)
not_bed = !(sphere_bed)

immersed_boundary = (;not_air, not_bed)

free_surface_bc = PrescribedTraction(0.0, 0.0, 0.0)
no_slip_wall_bc = PrescribedVelocity(0.0, 0.0, 0.0)

adiabatic_flux_bc    = PrescribedHeatFlux(0.0)
geothermal_flux_bc   = PrescribedHeatFlux(0.1)
fixed_temperature_bc = PrescribedTemperature(-0.1)

boundary_conditions = (
    stokes = BoundaryConditions(
        no_slip_wall_bc; top = free_surface_bc, immersed_ice_air = free_surface_bc
    ),
    thermal = BoundaryConditions(
        adiabatic_flux_bc; immersed_ice_air = fixed_temperature; immersed_ice_bed = geothermal_flux_bc
    )
)

## Thermodynamics
equation_of_state = IncompressibleEOS(
    density       = (ice = 920.0 , water = 1000.0),
    heat_capacity = (ice = 2100.0, water = 4200.0),
    conductivity  = (ice = 2.0   , water = 1.0),
    latent_heat   = 334e3
)

## Rheology
rheology = IceRheology(
    stress_strain     = Glen(A = 1e-20, n = 3),               # variants: Linear(μ), Goldsby(...)
    thermal_weakening = Arrhenius(Q = 1.0, R = 8.31),         # variants: nothing
    melt_weakening    = PowerLawWeakening(ϕref = 0.1, n = 2), # variants: nothing
)

physics = (;equation_of_state, rheology)

numerics = (
    tolerance              = (V = 1e-6, τ = 1e-6, Pr = 1e-8),
    check_after_iterations = 100,
    max_iterations         = 20(size(grid),3)
)

advection = UpwindAdvection() # variants: 

mass_balance = nothing

model = ThermomechanicalStokesModel(;
    grid,
    immersed_boundary,
    physics,
    advection,
    boundary_conditions,
    numerics,
)

## Initialisation
# Variants:
# 1. Constant
#   τ_ini = 0.0
# 2. Constant per-component
#   τ_ini = (
#     xx = 0.0, yy = 0.0, zz = 0.0,
#     xy = 0.0, xz = 0.0, yz = 0.0,
# )
# 3. Function of coordinates and time
#   Pr_ini(x,y,z) = ρg*(zc[end] - z)
# 4. Parametrized function
#   Pr_ini(x,y,z,p) = p.ρg*(p.lz - z)
# 5. Function of grid indices and time
#   Pr_ini(ix,iy,iz,grid) = ρg*(grid[ix,iy,end][3] - grid[ix,iy,iz][3])
# 6. Parametrized function of grid indices
#   Pr_ini(ix,iy,iz,grid,p) = p.ρg*(p.lz - grid[ix,iy,iz][3])

τ_ini = 0.0
P_ini = 0.0
v_ini = 0.0

ρU_ini = total_energy(equation_of_state, (T=-10.0, ϕ=0.0))

set!(model, τ = τ_ini, P = P_ini, v = v_ini, ρU = ρU_ini)

timestepping = (
    rep_Δt     = 1.0,
    total_time = 1000.0,
    min_Δt     = 0.01,
    max_Δt     = 0.5,
    cfl        = 1/3.1,
)

simulation = Simulation(model; timestepping...)

## High-level framework api
# callbacks = CallbackSet(...)
# run!(simulation; callbacks)
run!(simulation)

## Intermediate-level API
for (it, rep_Δt, current_time) in timesteps(simulation)
    copy_double_buffers!(model)
    target_Δt = timestepping.max_Δt
    if !isnothing(timestepping.cfl)
        cfl_Δt    = estimate_Δt(model, timestepping.cfl)
        target_Δt = min(target_Δt, cfl_Δt)
    end
    nsub = ceil(Int,rep_Δt/target_Δt)
    Δt   = rep_Δt/nsub
    isub = 0; Δt_stack = fill(Δt,nsub)
    while !isempty(Δt_stack)
        isub += 1
        Δt   = pop!(Δt_stack)
        if !advance_timestep!(model, Δt, current_time)
            recover!(model)
            push!(Δt_stack, 0.5Δt, 0.5Δt)
        else
            copy_double_buffers!(model)
            current_time += Δt
        end
    end
end

## Low-level library-like API
for (it, rep_Δt, current_time) in timesteps(simulation)
    copy_double_buffers!(model)
    target_Δt = timestepping.max_Δt
    if !isnothing(timestepping.cfl)
        cfl_Δt    = estimate_Δt(model, timestepping.cfl)
        target_Δt = min(target_Δt, cfl_Δt)
    end
    nsub = ceil(Int,rep_Δt/target_Δt)
    Δt   = rep_Δt/nsub
    isub = 0; Δt_stack = fill(Δt,nsub)
    while !isempty(Δt_stack)
        isub += 1
        Δt   = pop!(Δt_stack)
        ϵtol = Tuple(numerics.tolerance)
        iter = 1; errs = copy(ϵtol); finished = false; success = true
        while !finished
            advance_iteration!(model, Δt, current_time)
            iter += 1
            if iter % numerics.check_after_iterations == 0
                errs = compute_residual_norm(model)
                if !all(isfinite.(errs))
                    success  = false
                    finished = true
                else 
                    finished = all(errs .<= ϵtol)
                end
            end
            if iter > numerics.max_iterations
                success  = false
                finished = true
            end
        end
        if failed
            recover!(model)
            push!(Δt_stack, 0.5Δt, 0.5Δt)
        else
            copy_double_buffers!(model)
            current_time += Δt
        end
    end
end
