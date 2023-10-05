using FastIce.Utils

function main()
    pipes = [Pipeline() for _ in 1:6]
    for iter in 1:10
        for ip in eachindex(pipes)
            put!(pipes[ip]) do
                sleep(0.001)
                println("  inside pipeline #$(ip)!")
            end
        end
        sleep(0.01)
        println("outside, iter #$(iter)!")
        take!.(pipes)
        println()
    end
    setdone!.(pipes)
    return
end

main()
