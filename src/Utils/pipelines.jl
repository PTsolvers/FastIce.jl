mutable struct Pipeline
    in::Channel
    out::Base.Event
    task::Task

    function Pipeline(; pre=nothing, post=nothing)
        in = Channel()
        out = Base.Event(true)
        task = Threads.@spawn begin
            isnothing(pre) || pre()
            try
                for work in in
                    work()
                    notify(out)
                end
            finally
                isnothing(post) || post()
            end
        end
        errormonitor(task)
        return new(in, out, task)
    end
end

Base.close(p::Pipeline) = close(p.in)

Base.isopen(p::Pipeline) = isopen(p.in)

function Base.put!(work::F, p::Pipeline) where {F}
    put!(p.in, work)
    return
end

function Base.wait(p::Pipeline)
    if isopen(p.in)
        wait(p.out)
    else
        error("Pipeline is not running")
    end
    return
end
