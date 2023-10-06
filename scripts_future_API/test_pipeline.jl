using FastIce.Utils
using Profile

function dowork(pipes)
    for _ in 1:100
        for pipe in pipes
            put!(()->sleep(0.01), pipe)
        end
        wait.(pipes)
    end
    return
end

function main()
    pipes = Tuple(Pipeline() for _ in 1:4)
    dowork(pipes)
    dowork(pipes)
    close.(pipes)
    return
end

main()

