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
using AMDGPU

using FastIce.Distributed
using MPI

println("import done")

@views avx(A) = 0.5 .* (A[1:end-1, :, :] .+ A[2:end, :, :])
@views avy(A) = 0.5 .* (A[:, 1:end-1, :] .+ A[:, 2:end, :])
@views avz(A) = 0.5 .* (A[:, :, 1:end-1] .+ A[:, :, 2:end])

MPI.Init()

backend = ROCBackend()
dims    = (4, 2, 2)
arch    = Architecture(backend, dims, MPI.COMM_WORLD)

# physics
ebg = 1.0

topo = details(arch)

size_l = (254, 254, 254)
size_g = global_grid_size(topo, size_l)

if global_rank(topo) == 0
    @show dimensions(topo)
    @show size_g
end

grid_g = CartesianGrid(; origin=(-2.0, -1.0, 0.0),
                       extent=(4.0, 2.0, 2.0),
                       size=size_g)

grid_l = local_grid(grid_g, topo)

no_slip      = VBC(0.0, 0.0, 0.0)
free_slip    = SBC(0.0, 0.0, 0.0)
free_surface = TBC(0.0, 0.0, 0.0)

boundary_conditions = (x = (free_slip, free_slip),
                       y = (free_slip, free_slip),
                       z = (no_slip, free_surface))

gravity = (x=-0.25, y=0.0, z=1.0)

# numerics
nt     = 50maximum(size(grid_g))
ncheck = 1maximum(size(grid_g))

r       = 0.7
re_mech = 5π
lτ_re_m = minimum(extent(grid_g)) / re_mech
vdτ     = minimum(spacing(grid_g)) / sqrt(ndims(grid_g) * 3.1)
θ_dτ    = lτ_re_m * (r + 4 / 3) / vdτ
dτ_r    = 1.0 / (θ_dτ + 1.0)
nudτ    = vdτ * lτ_re_m

iter_params = (η_rel=1e-1,
               Δτ=(Pr=r / θ_dτ, τ=(xx=dτ_r, yy=dτ_r, zz=dτ_r, xy=dτ_r, xz=dτ_r, yz=dτ_r), V=(x=nudτ, y=nudτ, z=nudτ)))

physics = (rheology=GlensLawRheology(1),)
other_fields = (A=Field(backend, grid_l, Center()),)

model = IsothermalFullStokesModel(;
                                  arch,
                                  grid=grid_l,
                                  physics,
                                  gravity,
                                  boundary_conditions,
                                  iter_params,
                                  other_fields)

if global_rank(topo) == 0
    println("model created")
    Pr_g = zeros(size(grid_g))
    Vx_g = zeros(size(grid_g))
    Vy_g = zeros(size(grid_g))
    Vz_g = zeros(size(grid_g))
else
    Pr_g = nothing
    Vx_g = nothing
    Vy_g = nothing
    Vz_g = nothing
end

Pr_v = zeros(size(grid_l))
Vx_v = zeros(size(grid_l))
Vy_v = zeros(size(grid_l))
Vz_v = zeros(size(grid_l))

# set!(model.fields.Pr, 0.0)
# foreach(x -> set!(x, 0.0), model.fields.τ)

fill!(parent(model.fields.Pr), 0.0)
foreach(x -> fill!(parent(x), 0.0), model.fields.τ)
foreach(x -> fill!(parent(x), 0.0), model.fields.V)
fill!(parent(other_fields.A), 1.0)
set!(model.fields.η, other_fields.A)

KernelLaunch.apply_all_boundary_conditions!(arch, grid_l, model.boundary_conditions.stress)
KernelLaunch.apply_all_boundary_conditions!(arch, grid_l, model.boundary_conditions.velocity)
KernelLaunch.apply_all_boundary_conditions!(arch, grid_l, model.boundary_conditions.rheology)

if global_rank(topo) == 0
    println("action")
end

for it in 1:nt
    advance_iteration!(model, 0.0, 1.0; async=false)
    if (it % ncheck == 0) && (global_rank(topo) == 0)
        println("iter/nx = $(it/maximum(size(grid_g)))")
    end
end

comm = cartesian_communicator(topo)

copyto!(Pr_v, interior(model.fields.Pr))
copyto!(Vx_v, avx(interior(model.fields.V.x)))
copyto!(Vy_v, avy(interior(model.fields.V.y)))
copyto!(Vz_v, avz(interior(model.fields.V.z)))

KernelAbstractions.synchronize(backend)

gather!(Pr_g, Pr_v, comm)
gather!(Vx_g, Vx_v, comm)
gather!(Vy_g, Vy_v, comm)
gather!(Vz_g, Vz_v, comm)

if global_rank(topo) == 0
    open("data.bin", "w") do io
        write(io, Pr_g)
        write(io, Vx_g)
        write(io, Vy_g)
        write(io, Vz_g)
    end
end

MPI.Finalize()
