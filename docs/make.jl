using Documenter
using FastIce

push!(LOAD_PATH,"../src/")

makedocs(
    sitename = "FastIce",
    authors="Ludovic Räss, Ivan Utkin and contributors",
    format = Documenter.HTML(
        prettyurls=get(ENV, "CI", nothing) == "true", # easier local build
        ansicolor=true
        ),
    modules = [FastIce],
    warnonly = [:missing_docs],
    pages = Any[
        "Home" => "index.md",
        "Usage" => Any["usage/runtests.md"],
        "Library" => Any["lib/modules.md"]
    ]
)

deploydocs(
    repo = "github.com/PTsolvers/FastIce.jl.git",
    devbranch = "main",
    push_preview = true
)
