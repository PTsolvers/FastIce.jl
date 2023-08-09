include("common.jl")

using FastIce.Fields
using FastIce.Grids

const HALOS = ((1, 1, 2), nothing, (1, nothing, 1), (nothing, nothing, nothing), (nothing, (1, nothing), (1, 2)))
const HALO_SIZES = ((2, 2, 4), (0, 0, 0), (2, 0, 2), (0, 0, 0), (0, 1, 3))

@testset "backend $backend" for backend in backends
    @testset "fields" begin
        grid = CartesianGrid(origin = (0.0, 0.0, 0.0), extent=(1.0, 1.0, 1.0), size = (2, 2, 2))
        @testset "halo $hl" for (hl, hs) in zip(HALOS, HALO_SIZES)
            loc = (Center(), Vertex(), Center())
            f = Field(backend, grid, loc, hl)
            @test location(f) == (Center(), Vertex(), Center())
            @test size(f) == size(grid, loc) .+ hs
            @test size(interior(f)) == size(grid, loc)
        end
    end
end