using Documenter
using FastIce

push!(LOAD_PATH,"../src/")

DocMeta.setdocmeta!(FastIce, :DocTestSetup, :(using FastIce); recursive=true)

makedocs(
    sitename = "FastIce",
    authors="Ludovic Räss, Ivan Utkin and contributors",
    format = Documenter.HTML(
        prettyurls=get(ENV, "CI", nothing) == "true", # easier local build
        ansicolor=true
        ),
    modules = [FastIce],
    pages=[
        "Home" => "index.md",
        "Usage" => "usage.md",
        "Library" => "library.md"
    ]
)

deploydocs(
    repo = "github.com/PTsolvers/FastIce.jl.git",
    devbranch = "main"
)
