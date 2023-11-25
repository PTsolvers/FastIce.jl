module FastIce

const GREETING = raw"""
┌──────────────────────────────────────────────────────────┐
│     ______              __   ____                  _  __ │
│    / ____/____ _ _____ / /_ /  _/_____ ___        (_)/ / │
│   / /_   / __ `// ___// __/ / / / ___// _ \      / // /  │
│  / __/  / /_/ /(__  )/ /_ _/ / / /__ /  __/_    / // /   │
│ /_/     \__,_//____/ \__//___/ \___/ \___/(_)__/ //_/    │
│                                             /___/        │
└──────────────────────────────────────────────────────────┘
"""

greet(; kwargs...) = printstyled(GREETING; kwargs...)

# core modules
include("Grids/Grids.jl")
include("GridOperators.jl")
include("Logging.jl")
include("Architectures.jl")
include("Fields/Fields.jl")
include("Utils/Utils.jl")
include("BoundaryConditions/BoundaryConditions.jl")
include("KernelLaunch.jl")
include("Distributed/Distributed.jl")
include("Physics.jl")

# ice flow models
include("Models/Models.jl")

end # module
