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

vx(x, y, z) = sin(π * x) * (cos(π * y) * (z^2 - 1)^2 - cos(π * z) * (y^2 - 1)^2)
vy(x, y, z) = sin(π * y) * (cos(π * z) * (x^2 - 1)^2 - cos(π * x) * (z^2 - 1)^2)
vz(x, y, z) = sin(π * z) * (cos(π * x) * (y^2 - 1)^2 - cos(π * y) * (x^2 - 1)^2)

τxx(x, y, z, η) = 2 * η * π * cos(π * x) * (cos(π * y) * (z^2 - 1)^2 - cos(π * z) * (y^2 - 1)^2)
τyy(x, y, z, η) = 2 * η * π * cos(π * y) * (cos(π * z) * (x^2 - 1)^2 - cos(π * x) * (z^2 - 1)^2)
τzz(x, y, z, η) = 2 * η * π * cos(π * z) * (cos(π * x) * (y^2 - 1)^2 - cos(π * y) * (x^2 - 1)^2)
τxy(x, y, z, η) = 4 * η * cos(π * z) * ((x^3 - x) * sin(π * y) - (y^3 - y) * sin(π * x))
τxz(x, y, z, η) = 4 * η * cos(π * y) * ((z^3 - z) * sin(π * x) - (x^3 - x) * sin(π * z))
τyz(x, y, z, η) = 4 * η * cos(π * x) * ((y^3 - y) * sin(π * z) - (z^3 - z) * sin(π * y))

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
    r           = 0.7
    re_mech     = 5π
    lτ_re_m     = minimum(extent(grid_g)) / re_mech
    vdτ         = minimum(spacing(grid_g)) / sqrt(ndims(grid_g) * 10.1)
    θ_dτ        = lτ_re_m * (r + 4 / 3) / vdτ
    dτ_r        = 1.0 / (θ_dτ + 1.0)
    nudτ        = vdτ * lτ_re_m
    iter_params = (η_rel=1e-1,     # pack PT params
    Δτ=(Pr=r / θ_dτ, τ=(xx=dτ_r, yy=dτ_r, zz=dτ_r, xy=dτ_r, xz=dτ_r, yz=dτ_r), V=(x=nudτ, y=nudτ, z=nudτ)))
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
