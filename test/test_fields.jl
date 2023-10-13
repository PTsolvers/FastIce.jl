include("common.jl")

using FastIce.Fields
using FastIce.Grids

const HALOS = ((1, 1, 2), nothing, (1, nothing, 1), (nothing, nothing, nothing), (nothing, (1, nothing), (1, 2)))
const HALO_SIZES = ((2, 2, 4), (0, 0, 0), (2, 0, 2), (0, 0, 0), (0, 1, 3))

@testset "backend $backend" for backend in backends
    @testset "fields" begin
        grid = CartesianGrid(; origin=(0.0, 0.0, 0.0), extent=(1.0, 1.0, 1.0), size=(2, 2, 2))
        loc = (Center(), Vertex(), Center())
        @testset "location" begin
            @test location(Field(backend, grid, Center())) == (Center(), Center(), Center())
            @test location(Field(backend, grid, loc)) == loc
        end
        @testset "halo $hl" for (hl, hs) in zip(HALOS, HALO_SIZES)
            f = Field(backend, grid, loc; halo=hl)
            @test location(f) == (Center(), Vertex(), Center())
            @test size(data(f)) == size(grid, loc) .+ hs
            @test size(interior(f)) == size(grid, loc)
        end
        @testset "set" begin
            f = Field(backend, grid, (Center(), Vertex(), Center()); halo=(1, 0, 1))
            @testset "discrete" begin
                # no parameters vertex
                fill!(data(f), NaN)
                set!(f, grid, (grid, loc, ix, iy, iz) -> ycoord(grid, loc, iy); discrete=true)
                @test Array(interior(f)) == [0.0; 0.0;; 0.5; 0.5;; 1.0; 1.0;;;
                                             0.0; 0.0;; 0.5; 0.5;; 1.0; 1.0]
                # no parameters center
                fill!(data(f), NaN)
                set!(f, grid, (grid, loc, ix, iy, iz) -> xcoord(grid, loc, ix); discrete=true)
                @test Array(interior(f)) == [0.25; 0.75;; 0.25; 0.75;; 0.25; 0.75;;;
                                             0.25; 0.75;; 0.25; 0.75;; 0.25; 0.75]
                # with parameters
                fill!(data(f), NaN)
                set!(f, grid, (grid, loc, ix, iy, iz, sc) -> ycoord(grid, loc, iy) * sc; discrete=true, parameters=(2.0,))
                @test Array(interior(f)) == [0.0; 0.0;; 1.0; 1.0;; 2.0; 2.0;;;
                                             0.0; 0.0;; 1.0; 1.0;; 2.0; 2.0]
            end
            @testset "continuous" begin
                # no parameters vertex
                fill!(data(f), NaN)
                set!(f, grid, (x, y, z) -> y)
                @test Array(interior(f)) == [0.0; 0.0;; 0.5; 0.5;; 1.0; 1.0;;;
                                             0.0; 0.0;; 0.5; 0.5;; 1.0; 1.0]
                # no parameters center
                fill!(data(f), NaN)
                set!(f, grid, (x, y, z) -> x)
                @test Array(interior(f)) == [0.25; 0.75;; 0.25; 0.75;; 0.25; 0.75;;;
                                             0.25; 0.75;; 0.25; 0.75;; 0.25; 0.75]
                # with parameters
                fill!(data(f), NaN)
                set!(f, grid, (x, y, z, sc) -> y * sc; parameters=(2.0,))
                @test Array(interior(f)) == [0.0; 0.0;; 1.0; 1.0;; 2.0; 2.0;;;
                                             0.0; 0.0;; 1.0; 1.0;; 2.0; 2.0]
            end
        end
    end
end
