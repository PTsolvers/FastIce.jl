module FastIce

using KernelAbstractions

include("grid_operators.jl")
include("logging.jl")
include("grids.jl")
include("fields.jl")
include("utils.jl")

include("physics.jl")
include("boundary_conditions.jl")
include("models/models.jl")

end # module
