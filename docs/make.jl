using Documenter
using FastIce

push!(LOAD_PATH,"../src/")

const DOC_SETUP = quote
    using FastIce
    using FastIce.Grids
end

DocMeta.setdocmeta!(FastIce, :DocTestSetup, DOC_SETUP; recursive=true)

makedocs(
    sitename = "FastIce",
    authors="Ludovic Räss, Ivan Utkin and contributors",
    format = Documenter.HTML(
        prettyurls=get(ENV, "CI", nothing) == "true", # easier local build
        ansicolor=true
        ),
    modules = [FastIce],
    doctest = false,
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
