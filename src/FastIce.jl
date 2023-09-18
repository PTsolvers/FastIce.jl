module FastIce

using KernelAbstractions

include("Grids/Grids.jl")

include("grid_operators.jl")
include("logging.jl")
include("fields.jl")
include("utils.jl")

include("physics.jl")

include("BoundaryConditions/boundary_conditions.jl")
include("Models/models.jl")

# include("Distributed/distributed.jl")

end # module
