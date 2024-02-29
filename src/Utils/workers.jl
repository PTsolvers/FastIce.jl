mutable struct Worker
    src::Channel
    out::Base.Event
    task::Task

    function Worker(; setup=nothing, teardown=nothing)
        src = Channel()
        out = Base.Event(true)
        task = Threads.@spawn begin
            isnothing(setup) || Base.invokelatest(setup)
            try
                for work in src
                    Base.invokelatest(work)
                    notify(out)
                end
            finally
                isnothing(teardown) || Base.invokelatest(teardown)
            end
        end
        errormonitor(task)
        return new(src, out, task)
    end
end

function Base.close(p::Worker)
    close(p.src)
    wait(p.task)
    return
end

Base.isopen(p::Worker) = isopen(p.src)

function Base.put!(work::F, p::Worker) where {F}
    put!(p.src, work)
    return
end

function Base.wait(p::Worker)
    if isopen(p.src)
        wait(p.out)
    else
        error("Worker is not running")
    end
    return
end
