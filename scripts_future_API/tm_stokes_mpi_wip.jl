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

# using GLMakie

using FastIce.Distributed
using MPI

MPI.Init()

backend = CPU()
dims    = (2, 0, 2)
arch    = Architecture(backend, dims, MPI.COMM_WORLD)

# physics
ebg = 1.0

topo = details(arch)

size_l = (32, 32, 32)
size_g = global_grid_size(topo, size_l)

if global_rank(topo) == 0
    @show dimensions(topo)
    @show size_g
end

grid_g = CartesianGrid(; origin=(-1.0, -0.5, 0.0),
                       extent=(2.0, 1.0, 2.0),
                       size=size_g)

grid_l = local_grid(grid_g, topo)

psh_x(x, _, _) = -x * ebg
psh_y(_, y, _) = y * ebg

x_bc = BoundaryFunction(psh_x; reduce_dims=false)
y_bc = BoundaryFunction(psh_y; reduce_dims=false)

boundary_conditions = (x = (VBC(x_bc, y_bc, 0.0), VBC(x_bc, y_bc, 0.0)),
                       y = (VBC(x_bc, y_bc, 0.0), VBC(x_bc, y_bc, 0.0)),
                       z = (SBC(0.0, 0.0, 0.0), TBC(0.0, 0.0, 0.0)))

# numerics
nt   = 1000
nviz = 10

r       = 0.7
re_mech = 10π
lτ_re_m = minimum(extent(grid_g)) / re_mech
vdτ     = minimum(spacing(grid_g)) / sqrt(ndims(grid_g) * 3.1)
θ_dτ    = lτ_re_m * (r + 4 / 3) / vdτ
dτ_r    = 1.0 / (θ_dτ + 1.0)
nudτ    = vdτ * lτ_re_m

iter_params = (η_rel=1e-1,
               Δτ=(Pr=r / θ_dτ, τ=(xx=dτ_r, yy=dτ_r, zz=dτ_r, xy=dτ_r, xz=dτ_r, yz=dτ_r), V=(x=nudτ, y=nudτ, z=nudτ)))

physics = (rheology=GlensLawRheology(1),)
other_fields = (A=Field(backend, grid_l, Center()),)

init_incl(x, y, z, x0, y0, z0, r, Ai, Am) = ifelse((x - x0)^2 + (y - y0)^2 + (z - z0)^2 < r^2, Ai, Am)
set!(other_fields.A, grid_l, init_incl; parameters=(x0=0.0, y0=0.0, z0=0.5, r=0.2, Ai=1e-1, Am=1.0))

model = IsothermalFullStokesModel(;
                                  arch,
                                  grid=grid_l,
                                  physics,
                                  boundary_conditions,
                                  iter_params,
                                  other_fields)

# if global_rank(topo) == 0
    # fig = Figure(; resolution=(1200, 1000), fontsize=32)
    # axs = (Pr=Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="z", title="Pr"),
    #        Vx=Axis(fig[1, 2][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="z", title="Vx"),
    #        Vy=Axis(fig[2, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="z", title="Vy"),
    #        Vz=Axis(fig[2, 2][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="z", title="Vz"))

    # plt = (Pr=heatmap!(axs.Pr, xcenters(grid_l), zcenters(grid_l), interior(model.fields.Pr)[:, size(grid_l, 2)÷2, :]; colormap=:turbo),
    #        Vx=heatmap!(axs.Vx, xvertices(grid_l), zcenters(grid_l), interior(model.fields.V.x)[:, size(grid_l, 2)÷2, :]; colormap=:turbo),
    #        Vy=heatmap!(axs.Vy, xcenters(grid_l), zcenters(grid_l), interior(model.fields.V.y)[:, size(grid_l, 2)÷2, :]; colormap=:turbo),
    #        Vz=heatmap!(axs.Vz, xcenters(grid_l), zvertices(grid_l), interior(model.fields.V.z)[:, size(grid_l, 2)÷2, :]; colormap=:turbo))
    # Colorbar(fig[1, 1][1, 2], plt.Pr)
    # Colorbar(fig[1, 2][1, 2], plt.Vx)
    # Colorbar(fig[2, 1][1, 2], plt.Vy)
    # Colorbar(fig[2, 2][1, 2], plt.Vz)
# end

# set!(model.fields.Pr, 0.0)
# foreach(x -> set!(x, 0.0), model.fields.τ)

fill!(parent(model.fields.Pr), 0.0)
foreach(x -> fill!(parent(x), 0.0), model.fields.τ)
foreach(x -> fill!(parent(x), 0.0), model.fields.V)

KernelLaunch.apply_all_boundary_conditions!(arch, grid_l, model.boundary_conditions.stress)
println("$(global_rank(topo)) applied stress BCs")

set!(model.fields.V.x, grid_l, psh_x)
set!(model.fields.V.y, grid_l, psh_y)
set!(model.fields.V.z, 0.0)

# println("at rank $(global_rank(topo)) bcs $(typeof(model.boundary_conditions.velocity))")
println("at rank $(global_rank(topo)) topo $topo")
MPI.Barrier(cartesian_communicator(topo))

KernelLaunch.apply_all_boundary_conditions!(arch, grid_l, model.boundary_conditions.velocity)
println("$(global_rank(topo)) applied velocity BCs")

set!(model.fields.η, other_fields.A)
extrapolate!(model.fields.η)

# if global_rank(topo) == 0
    # display(fig)
# end

println("Hi from $(global_rank(topo))")
MPI.Barrier(cartesian_communicator(topo))

for it in 1:nt
    advance_iteration!(model, 0.0, 1.0; async=false)
    if it % nviz == 0# && global_rank(topo) == 0
        if global_rank(topo) == 0
            println("it = $it/$nt")
        end
        # plt.Pr[3][] = interior(model.fields.Pr)[:, size(grid_l, 2)÷2, :]
        # plt.Vx[3][] = interior(model.fields.V.x)[:, size(grid_l, 2)÷2, :]
        # plt.Vy[3][] = interior(model.fields.V.y)[:, size(grid_l, 2)÷2, :]
        # plt.Vz[3][] = interior(model.fields.V.z)[:, size(grid_l, 2)÷2, :]
        # yield()
    end
end

sleep(30)

MPI.Finalize()
