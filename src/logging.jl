module Logging

export MPILogger

import Logging: AbstractLogger, handle_message, shouldlog, min_enabled_level
import MPI

struct MPILogger{B<:AbstractLogger} <: AbstractLogger
    rank::Int64
    comm::MPI.Comm
    base_logger::B
end

function handle_message(l::MPILogger, args...; kwargs...)
    if MPI.Comm_rank(l.comm) == l.rank
        handle_message(l.base_logger, args...; kwargs...)
    end
end

shouldlog(l::MPILogger, args...) = (MPI.Comm_rank(l.comm) == l.rank) && shouldlog(l.base_logger, args...)

min_enabled_level(l::MPILogger) = min_enabled_level(l.base_logger)

end
