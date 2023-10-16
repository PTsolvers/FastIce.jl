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
    f[I] = val + I[1]
end

MPI.Init()

arch = Architecture(CPU(), (0, 0))
grid = CartesianGrid(; origin=(0.0, 0.0), extent=(1.0, 1.0), size=(10, 10))
field = Field(backend(arch), grid, (Vertex(), Vertex()); halo=1)


me = global_rank(details(arch))

fill!(parent(field), Inf)

bc = FieldBoundaryConditions((field,), (DirichletBC{FullCell}(-me-10),))

boundary_conditions = ((bc, bc),
                       (bc, bc))

boundary_conditions = ntuple(Val(length(boundary_conditions))) do D
    ntuple(Val(2)) do S
        if neighbor(details(arch), D, S) != MPI.PROC_NULL
            DistributedBoundaryConditions(Val(S), Val(D), (field, ))
        else
            boundary_conditions[D][S]
        end
    end
end

hide_boundaries = HideBoundaries{2}(arch)

outer_width = (4, 4)

launch!(arch, grid, fill_field! => (field, me); location=location(field), hide_boundaries, boundary_conditions, outer_width)

sleep(me)
@show coordinates(details(arch))
display(parent(field))

MPI.Finalize()
