module FastIce

using KernelAbstractions

include("Grids/Grids.jl")

include("GridOperators.jl")
include("Logging.jl")
include("Fields.jl")
include("Architectures.jl")

include("Utils/Utils.jl")

include("Physics.jl")

include("BoundaryConditions/BoundaryConditions.jl")
include("Models/models.jl")

include("Distributed/Distributed.jl")

end # module
