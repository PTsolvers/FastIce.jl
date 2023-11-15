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

using CairoMakie

using FastIce.Distributed
using MPI

println("import done")

@views avx(A) = 0.5 .* (A[1:end-1, :, :] .+ A[2:end, :, :])
@views avy(A) = 0.5 .* (A[:, 1:end-1, :] .+ A[:, 2:end, :])
@views avz(A) = 0.5 .* (A[:, :, 1:end-1] .+ A[:, :, 2:end])

MPI.Init()

backend = ROCBackend()
dims    = (0, 0, 0)
arch    = Architecture(backend, dims, MPI.COMM_WORLD)

# physics
ebg = 1.0

topo = details(arch)

size_l = (126, 126, 126)
size_g = global_grid_size(topo, size_l)

if global_rank(topo) == 0
    @show dimensions(topo)
    @show size_g
end

grid_g = CartesianGrid(; origin=(-1.0, -1.0, 0.0),
                       extent=(2.0, 2.0, 2.0),
                       size=size_g)

grid_l = local_grid(grid_g, topo)

psh_x(x, _, _, ebg) = -x * ebg
psh_y(_, y, _, ebg) = y * ebg

x_bc = BoundaryFunction(psh_x; reduce_dims=false, parameters=(ebg,))
y_bc = BoundaryFunction(psh_y; reduce_dims=false, parameters=(ebg,))

boundary_conditions = (x = (VBC(x_bc, y_bc, 0.0), VBC(x_bc, y_bc, 0.0)),
                       y = (VBC(x_bc, y_bc, 0.0), VBC(x_bc, y_bc, 0.0)),
                       z = (SBC(0.0, 0.0, 0.0), TBC(0.0, 0.0, 0.0)))

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

init_incl(x, y, z, x0, y0, z0, r, Ai, Am) = ifelse((x - x0)^2 + (y - y0)^2 + (z - z0)^2 < r^2, Ai, Am)
set!(other_fields.A, grid_l, init_incl; parameters=(x0=0.0, y0=0.0, z0=1.0, r=0.2, Ai=1e-1, Am=1.0))

model = IsothermalFullStokesModel(;
                                  arch,
                                  grid=grid_l,
                                  physics,
                                  boundary_conditions,
                                  iter_params,
                                  other_fields)

println("model created")

if global_rank(topo) == 0
    fig = Figure(; resolution=(1200, 1000), fontsize=32)
    axs = (Pr=Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="z", title="Pr"),
           Vx=Axis(fig[1, 2][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="z", title="Vx"),
           Vy=Axis(fig[2, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="z", title="Vy"),
           Vz=Axis(fig[2, 2][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="z", title="Vz"))

    Pr_g = zeros(size(grid_g))
    Vx_g = zeros(size(grid_g))
    Vy_g = zeros(size(grid_g))
    Vz_g = zeros(size(grid_g))

    plt = (Pr=heatmap!(axs.Pr, xcenters(grid_g), zcenters(grid_g), @view Pr_g[:, size(grid_g, 2)÷2, :]; colormap=:turbo),
           Vx=heatmap!(axs.Vx, xcenters(grid_g), zcenters(grid_g), @view Vx_g[:, size(grid_g, 2)÷2, :]; colormap=:turbo),
           Vy=heatmap!(axs.Vy, xcenters(grid_g), zcenters(grid_g), @view Vy_g[:, size(grid_g, 2)÷2, :]; colormap=:turbo),
           Vz=heatmap!(axs.Vz, xcenters(grid_g), zcenters(grid_g), @view Vz_g[:, size(grid_g, 2)÷2, :]; colormap=:turbo))
    Colorbar(fig[1, 1][1, 2], plt.Pr)
    Colorbar(fig[1, 2][1, 2], plt.Vx)
    Colorbar(fig[2, 1][1, 2], plt.Vy)
    Colorbar(fig[2, 2][1, 2], plt.Vz)
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

KernelLaunch.apply_all_boundary_conditions!(arch, grid_l, model.boundary_conditions.stress)

set!(model.fields.V.x, grid_l, psh_x; parameters=(ebg,))
set!(model.fields.V.y, grid_l, psh_y; parameters=(ebg,))
set!(model.fields.V.z, 0.0)

KernelLaunch.apply_all_boundary_conditions!(arch, grid_l, model.boundary_conditions.velocity)

set!(model.fields.η, other_fields.A)
KernelLaunch.apply_all_boundary_conditions!(arch, grid_l, model.boundary_conditions.rheology)

if global_rank(topo) == 0
    display(fig)
end

@info "action"

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
    plt.Pr[3][] = @view Pr_g[:, size(grid_g, 2)÷2, :]
    plt.Vx[3][] = @view Vx_g[:, size(grid_g, 2)÷2, :]
    plt.Vy[3][] = @view Vy_g[:, size(grid_g, 2)÷2, :]
    plt.Vz[3][] = @view Vz_g[:, size(grid_g, 2)÷2, :]

    save("stokes.png", fig)
end

MPI.Finalize()
