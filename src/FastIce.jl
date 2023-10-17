module FastIce

# core modules
include("Grids/Grids.jl")
include("GridOperators.jl")
include("Logging.jl")
include("Architectures.jl")
include("Fields.jl")
include("Utils/Utils.jl")
include("BoundaryConditions/BoundaryConditions.jl")
include("KernelLaunch.jl")
include("Distributed/Distributed.jl")
include("Physics.jl")

# ice flow models
include("Models/models.jl")

end # module
