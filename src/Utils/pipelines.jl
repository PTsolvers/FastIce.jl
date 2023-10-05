mutable struct Pipeline
    @atomic done::Bool
    in::Channel
    out::Base.Event
    task::Task

    function Pipeline(; pre=nothing, post=nothing)
        in = Channel()
        out = Base.Event(true)
        this = new(false, in, out)
    
        this.task = Threads.@spawn begin
            isnothing(pre) || pre()
            try
                while !(@atomic this.done)
                    work = take!(in)
                    work()
                    notify(out)
                end
            catch err
                @atomic this.done = true
                rethrow(err)
            finally
                isnothing(post) || post()
            end
        end
        errormonitor(this.task)
        return this
    end
end

setdone!(p::Pipeline) = @atomic p.done = true

Base.isdone(p::Pipeline) = @atomic p.done

function Base.put!(work::F, p::Pipeline) where {F}
    if !(@atomic p.done)
        put!(p.in, work)
    else
        error("Pipeline is not running")
    end
    return
end

function Base.take!(p::Pipeline)
    if !(@atomic p.done)
        wait(p.out)
    else
        error("Pipeline is not running")
    end
    return
end
