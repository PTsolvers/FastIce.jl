module FastIce

const GREETING = raw"""
        ______           __  ____              _ __
       / ____/___ ______/ /_/  _/_______      (_) /
      / /_  / __ `/ ___/ __// // ___/ _ \    / / /
     / __/ / /_/ (__  ) /__/ // /__/  __/   / / /
    /_/    \__,_/____/\__/___/\___/\___(_)_/ /_/
                                        /___/
https://github.com/PTsolvers/FastIce.jl

"""

const GREETING_FAST = raw"""
    __________             _____________             ______________
    ___  ____/_____ _________  /____  _/__________   ______(_)__  /
    __  /_   _  __ `/_  ___/  __/__  / _  ___/  _ \  _____  /__  /
    _  __/   / /_/ /_(__  )/ /_ __/ /  / /__ /  __/______  / _  /
    /_/      \__,_/ /____/ \__/ /___/  \___/ \___/_(_)__  /  /_/
                                                     /___/
https://github.com/PTsolvers/FastIce.jl

"""

greet(; kwargs...) = printstyled(GREETING; kwargs...)
greet_fast(; kwargs...) = printstyled(GREETING_FAST; kwargs...)

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
include("Writers.jl")
include("LevelSets/LevelSets.jl")

# ice flow models
include("Models/Models.jl")

end # module
