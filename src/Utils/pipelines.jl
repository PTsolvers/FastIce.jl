mutable struct Pipeline
    src::Channel
    out::Base.Event
    task::Task

    function Pipeline(; pre=nothing, post=nothing)
        src = Channel()
        out = Base.Event(true)
        task = Threads.@spawn begin
            isnothing(pre) || Base.invokelatest(pre)
            try
                for work in src
                    Base.invokelatest(work)
                    notify(out)
                end
            finally
                isnothing(post) || Base.invokelatest(post)
            end
        end
        errormonitor(task)
        return new(src, out, task)
    end
end

function Base.close(p::Pipeline)
    close(p.src)
    wait(p.task)
    return
end

Base.isopen(p::Pipeline) = isopen(p.src)

function Base.put!(work::F, p::Pipeline) where {F}
    put!(p.src, work)
    return
end

function Base.wait(p::Pipeline)
    if isopen(p.src)
        wait(p.out)
    else
        error("Pipeline is not running")
    end
    return
end
