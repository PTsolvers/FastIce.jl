using FastIce
using FastIce.Architectures
using FastIce.Grids
using FastIce.Fields
using FastIce.Utils
using FastIce.BoundaryConditions
using FastIce.Models.FullStokes.Isothermal
using FastIce.Physics
using FastIce.KernelLaunch

const VBC = BoundaryCondition{Velocity}
const TBC = BoundaryCondition{Traction}
const SBC = BoundaryCondition{Slip}

using KernelAbstractions
# using AMDGPU

using FastIce.Distributed
using MPI

# manufactured solution for the confined Stokes flow with free-slip boundaries
# helper functions
f(ξ, η) = cos(π * ξ) * (η^2 - 1)^2 -
          cos(π * η) * (ξ^2 - 1)^2
g(ξ, η) = sin(π * η) * (ξ^2 - 1) * ξ -
          sin(π * ξ) * (η^2 - 1) * η
p(ξ, η) = cos(π * η) * (1 - 3 * ξ^2) * 2 -
          cos(π * ξ) * (1 - 3 * η^2) * 2
# velocity
vx(x, y, z) = sin(π * x) * f(y, z)
vy(x, y, z) = sin(π * y) * f(z, x)
vz(x, y, z) = sin(π * z) * f(x, y)
# diagonal deviatoric stress
τxx(x, y, z, η) = 2 * η * π * cos(π * x) * f(y, z)
τyy(x, y, z, η) = 2 * η * π * cos(π * y) * f(z, x)
τzz(x, y, z, η) = 2 * η * π * cos(π * z) * f(x, y)
# off-diagonal deviatoric stress
τxy(x, y, z, η) = 4 * η * cos(π * z) * g(x, y)
τxz(x, y, z, η) = 4 * η * cos(π * y) * g(z, x)
τyz(x, y, z, η) = 4 * η * cos(π * x) * g(y, z)
# forcing terms
ρgx(x, y, z, η) = -2 * η * sin(π * x) * (f(y, z) * π^2 - p(y, z))
ρgy(x, y, z, η) = -2 * η * sin(π * y) * (f(z, x) * π^2 - p(z, x))
ρgz(x, y, z, η) = -2 * η * sin(π * z) * (f(x, y) * π^2 - p(x, y))

function main()
    MPI.Init()
    # architecture
    backend = CPU()
    dims = (1, 1, 1)
    arch = Architecture(backend, dims, MPI.COMM_WORLD)
    set_device!(arch)
    topo = details(arch)
    # geometry
    size_l = (64, 64, 64) # local grid size
    size_g = global_grid_size(topo, size_l)
    if global_rank(topo) == 0
        @show dimensions(topo)
        @show size_g
    end
    grid_g = CartesianGrid(; origin=(-1.0, -1.0, -1.0),
                           extent=(2.0, 2.0, 2.0),
                           size=size_g)

    grid_l = local_grid(grid_g, topo)
    # physics
    η0 = 1.0
    A0 = 0.5
    # boundary conditions
    free_slip = SBC(0.0, 0.0, 0.0)
    boundary_conditions = (x = (free_slip, free_slip),
                           y = (free_slip, free_slip),
                           z = (free_slip, free_slip))
    gravity = (x=0.0, y=0.0, z=0.0)
    physics = (rheology=GlensLawRheology(1),)
    other_fields = (A=Field(backend, grid_l, Center()),)
    # numerics
    niter  = 20maximum(size(grid_g))
    ncheck = 1maximum(size(grid_g))
    # PT params
    r       = 0.7
    re_mech = 5π
    lτ_re_m = minimum(extent(grid_g)) / re_mech
    vdτ     = minimum(spacing(grid_g)) / sqrt(ndims(grid_g) * 10.1)
    θ_dτ    = lτ_re_m * (r + 4 / 3) / vdτ
    dτ_r    = 1.0 / (θ_dτ + 1.0)
    nudτ    = vdτ * lτ_re_m
    # pack PT params
    iter_params = (η_rel=1e-1, Δτ=(Pr=r / θ_dτ, τ=(xx=dτ_r, yy=dτ_r, zz=dτ_r, xy=dτ_r, xz=dτ_r, yz=dτ_r), V=(x=nudτ, y=nudτ, z=nudτ)))
    # model
    model = IsothermalFullStokesModel(; arch,
                                      grid=grid_l,
                                      physics,
                                      gravity,
                                      boundary_conditions,
                                      iter_params,
                                      other_fields)
    # init
    fill!(parent(model.fields.Pr), 0.0)
    foreach(x -> fill!(parent(x), 0.0), model.fields.τ)
    foreach(x -> fill!(parent(x), 0.0), model.fields.V)
    fill!(parent(other_fields.A), A0)
    set!(model.fields.η, grid_l, (grid, loc, I, fields) -> physics.rheology(grid, I, fields); discrete=true, parameters=(model.fields,))

    KernelLaunch.apply_all_boundary_conditions!(arch, grid_l, model.boundary_conditions.stress)
    KernelLaunch.apply_all_boundary_conditions!(arch, grid_l, model.boundary_conditions.velocity)
    KernelLaunch.apply_all_boundary_conditions!(arch, grid_l, model.boundary_conditions.rheology)

    if global_rank(topo) == 0
        println("action")
    end

    for iter in 1:niter
        advance_iteration!(model, 0.0, 1.0; async=false)
        if (iter % ncheck == 0) && (global_rank(topo) == 0)
            println("iter/nx = $(iter/maximum(size(grid_g)))")
        end
    end

    KernelAbstractions.synchronize(backend)
    MPI.Finalize()
    return
end

main()
