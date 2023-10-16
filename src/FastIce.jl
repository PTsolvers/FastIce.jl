module FastIce

include("Grids/Grids.jl")
include("GridOperators.jl")
include("Logging.jl")
include("Fields.jl")
include("Architectures.jl")
include("Utils/Utils.jl")
include("BoundaryConditions/BoundaryConditions.jl")
include("Models/models.jl")
include("KernelLaunch.jl")
include("Distributed/Distributed.jl")
include("Physics.jl")

end # module
