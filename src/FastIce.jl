module FastIce

using KernelAbstractions

include("utils.jl")
include("macros.jl")
include("logging.jl")
include("grids.jl")
include("fields.jl")

include("physics.jl")
include("boundary_conditions.jl")
include("models/models.jl")

end # module
