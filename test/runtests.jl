using Test
using FastIce

excludedfiles = [ "test_excluded.jl"];

function runtests()
    exename   = joinpath(Sys.BINDIR, Base.julia_exename())
    testdir   = pwd()
    istest(f) = endswith(f, ".jl") && startswith(basename(f), "test_")
    testfiles = sort(filter(istest, vcat([joinpath.(root, files) for (root, dirs, files) in walkdir(testdir)]...)))

    nfail = 0
    printstyled("Testing package FastIce.jl\n"; bold=true, color=:white)

    for f in testfiles
        println("")
        if f ∈ excludedfiles
            println("Test Skip:")
            println("$f")
            continue
        end
        try
            run(`$exename --startup-file=no $(joinpath(testdir, f))`)
        catch ex
            nfail += 1
        end
    end
    return nfail
end

exit(runtests())