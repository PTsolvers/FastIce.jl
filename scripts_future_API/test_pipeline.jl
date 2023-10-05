using FastIce.Utils
using Profile

function main()
    pipes = [Pipeline() for _ in 1:6]
    for iter in 1:100
        for ip in eachindex(pipes)
            put!(pipes[ip]) do
                sleep(1/60)
                # println("  inside pipeline #$(ip)!")
            end
        end
        sleep(1/10)
        # println("outside, iter #$(iter)!")
        take!.(pipes)
        # println()
    end
    setdone!(pipe)
    return
end

main()
