using Chmy, Chmy.Architectures, Chmy.Grids, Chmy.GridOperators, Chmy.Fields, Chmy.KernelLaunch, Chmy.BoundaryConditions
using FastIce.LevelSets
using KernelAbstractions

using GLMakie

# function main()
arch = Arch(CPU())
grid = UniformGrid(arch; origin=(-1, -1, -1), extent=(2, 2, 2), dims=(100, 100, 100))

grid_2D = UniformGrid(arch; origin=(-1, -1), extent=(2, 2), dims=(200, 200))


# bed parameters
amp = 0.05
ω   = 10π / 2.0
αx  = -tan(π / 15)
αy  = -tan(π / 15)

# ice parameters
x0  = -0.1
y0  = -0.1
z0  = -0.4
rad = 1.0

bed = FunctionField(grid_2D, Vertex(); parameters=(amp, ω, αx, αy)) do x, y, amp, ω, αx, αy
    return x * αx + y * αy + amp * sin(ω * x) * sin(ω * y)
end

ice = FunctionField(grid_2D, Vertex(); parameters=(x0, y0, z0, rad)) do x, y, x0, y0, z0, rad
    return z0 + sqrt(max(rad^2 - (x - x0)^2 - (y - y0)^2, 0.0))
end

ω = (na = FieldMask(arch, grid),
     ns = FieldMask(arch, grid))

ψ = (na = Field(arch, grid, Vertex()),
     ns = Field(arch, grid, Vertex()))

launch = Launcher(arch, grid)

compute_levelset_from_dem!(arch, launch, ψ.na, ice, grid_2D, grid)
compute_levelset_from_dem!(arch, launch, ψ.ns, bed, grid_2D, grid)

invert_levelset!(arch, launch, ψ.ns, grid)

ω_from_ψ!(arch, launch, ω.ns, ψ.ns, grid)
ω_from_ψ!(arch, launch, ω.na, ψ.na, grid)

# fig = Figure()
# ax  = Axis3(fig[1, 1]; aspect=:data)
# surface!(ax, vertices(grid_2D)..., interior(bed))
# surface!(ax, vertices(grid_2D)..., interior(ice))

# volume!(ax, vertices(grid)..., interior(ω.na.vvv); algorithm=:iso, isovalue=0.5, isorange=0.2)
# volume!(ax, vertices(grid)..., interior(ω.ns.vvv); algorithm=:iso, isovalue=0.5, isorange=0.2)

#     return
# end



# main()
