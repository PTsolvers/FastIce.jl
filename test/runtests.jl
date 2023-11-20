using Test
using FastIce

using Pkg

excludedfiles = ["test_excluded.jl"]

# distributed
test_distributed = ["test_distributed_2D.jl", "test_distributed_3D.jl"]
nprocs_2D = 4
nprocs_3D = 8
ENV["DIMX"] = 2
ENV["DIMY"] = 2
ENV["DIMZ"] = 2

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
        if basename(f) ∈ excludedfiles
            println("Test Skip:")
            println("$f")
            continue
        end
        try
            if basename(f) ∈ test_distributed
                nprocs = contains(f, "2D") ? nprocs_2D : nprocs_3D
                cmd(n=nprocs) = `mpiexecjl -n $n $exename --startup-file=no --color=yes $(joinpath(testdir, f))`
                withenv("JULIA_NUM_THREADS" => "4") do
                    run(cmd())
                end
            else
                run(`$exename --startup-file=no $(joinpath(testdir, f))`)
            end
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
