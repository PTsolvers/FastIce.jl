module FastIce

using KernelAbstractions

include("logging.jl")
include("grids.jl")
include("fields.jl")

include("physics.jl")
include("boundary_conditions.jl")

end # module
