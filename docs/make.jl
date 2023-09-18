using Documenter
using FastIce

push!(LOAD_PATH,"../src/")

makedocs(
    sitename = "FastIce",
    authors="Ludovic Räss, Ivan utkin and contributors",
    format = Documenter.HTML(; prettyurls=get(ENV, "CI", nothing) == "true"), # easier local build
    modules = [FastIce],
    pages=[
        "Home" => "index.md",
    ]
)

deploydocs(
    repo = "github.com/PTsolvers/FastIce.jl.git",
    devbranch = "main"
)
