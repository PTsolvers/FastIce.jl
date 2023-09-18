using Test
using FastIce

using Pkg

excludedfiles = [ "test_excluded.jl"];

function parse_flags!(args, flag; default=nothing, typ=typeof(default))
    for f in args
        startswith(f, flag) || continue

        if f != flag
            val = split(f, '=')[2]
            if !(typ ≡ nothing || typ <: AbstractString)
                @show typ val
                val = parse(typ, val)
            end
        else
            val = default
        end

        filter!(x -> x != f, args)
        return true, val
    end
    return false, default
end

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

_, backend_name = parse_flags!(ARGS, "--backend"; default="CPU", typ=String)

@static if backend_name == "AMDGPU"
    Pkg.add("AMDGPU")
    ENV["JULIA_FASTICE_BACKEND"] = "AMDGPU"
elseif backend_name == "CUDA"
    Pkg.add("CUDA")
    ENV["JULIA_FASTICE_BACKEND"] = "CUDA"
end

exit(runtests())