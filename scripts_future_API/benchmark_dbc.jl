using FastIce.Architectures
using FastIce.Distributed
using FastIce.Fields
using FastIce.Grids
using FastIce.BoundaryConditions
using FastIce.KernelLaunch

using KernelAbstractions
using MPI

@kernel function fill_field!(f, val, offset=nothing)
    I = @index(Global, Cartesian)
    if !isnothing(offset)
        I += offset
    end
    f[I] = val
end

MPI.Init()

arch = Architecture(CPU(), (2, 2, 2))
grid = CartesianGrid(; origin=(0.0, 0.0, 0.0), extent=(1.0, 1.0, 1.0), size=(5, 7, 5))
field = Field(backend(arch), grid, (Center(), Center(), Center()); halo=1)

me = global_rank(details(arch))

fill!(parent(field), Inf)

bc = BoundaryConditionsBatch((field,), (DirichletBC{FullCell}(-me-10),))

boundary_conditions = override_boundary_conditions(arch, ((bc, bc), (bc, bc), (bc, bc)); exchange=true)

hide_boundaries = HideBoundaries{3}(arch)

outer_width = (2, 2, 2)

launch!(arch, grid, fill_field! => (field, me); location=location(field), hide_boundaries, boundary_conditions, outer_width)

# sleep(0.25me)
# @show coordinates(details(arch))
# display(parent(field))

field_g = if global_rank(details(arch)) == 0
    KernelAbstractions.allocate(Architectures.backend(arch), eltype(field), dimensions(details(arch)) .* size(field))
else
    nothing
end

gather!(arch, field_g, field)

if global_rank(details(arch)) == 0
    println("global matrix:")
    display(field_g)
end

MPI.Finalize()
