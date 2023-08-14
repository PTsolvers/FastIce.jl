include("common.jl")

using FastIce.Grids
using FastIce.Fields
using FastIce.BoundaryConditions

@testset "backend $backend" for backend in backends
    @testset "boundary conditions" begin
        nx, ny, nz = 2, 2, 2
        grid  = CartesianGrid(origin = (0.0, 0.0, 0.0), extent = (1.0, 1.0, 1.0), size = (nx, ny, nz))
        field = Field(backend, grid, Center(); halo=1)
        @testset "value" begin
            @testset "x-dim" begin
                data(field) .= 0.0
                west_bc = DirichletBC{HalfCell}(1.0)
                east_bc = DirichletBC{FullCell}(0.5)
                discrete_bcs_x!(backend, 256, size(grid))(grid, (field, ), (west_bc, ), (east_bc, ))
                KernelAbstractions.synchronize(backend)
                @test all((@views field[0   , 1:ny, 1:nz] .+ field[1, 1:ny, 1:nz]) ./ 2 .≈ west_bc.val)
                @test all( @views field[nx+1, 1:ny, 1:nz] .≈ east_bc.val)
            end
            @testset "y-dim" begin
                data(field) .= 0.0
                south_bc = DirichletBC{HalfCell}(1.0)
                north_bc = DirichletBC{FullCell}(0.5)
                discrete_bcs_y!(backend, 256, size(grid))(grid, (field, ), (south_bc, ), (north_bc, ))
                KernelAbstractions.synchronize(backend)
                @test all((@views field[1:nx, 0   , 1:nz] .+ field[1:nx, 1, 1:nz]) ./ 2 .≈ south_bc.val)
                @test all( @views field[1:nx, ny+1, 1:nz] .≈ north_bc.val)
            end
            @testset "z-dim" begin
                data(field) .= 0.0
                bot_bc = DirichletBC{HalfCell}(1.0)
                top_bc = DirichletBC{FullCell}(0.5)
                discrete_bcs_z!(backend, 256, size(grid))(grid, (field, ), (bot_bc, ), (top_bc, ))
                KernelAbstractions.synchronize(backend)
                @test all((@views field[1:nx, 1:ny, 0   ] .+ field[1:nx, 1:ny, 2]) ./ 2 .≈ bot_bc.val)
                @test all( @views field[1:nx, 1:ny, nz+1] .≈ top_bc.val)
            end
        end
        @testset "array" begin
            @testset "x-dim" begin
                data(field) .= 0.0
                bc_array_west = KernelAbstractions.allocate(backend, Float64, (size(grid, 2), size(grid, 3)))
                bc_array_east = KernelAbstractions.allocate(backend, Float64, (size(grid, 2), size(grid, 3)))
                bc_array_west .= 1.0
                bc_array_east .= 0.5
                west_bc = DirichletBC{HalfCell}(bc_array_west)
                east_bc = DirichletBC{FullCell}(bc_array_east)
                discrete_bcs_x!(backend, 256, size(grid))(grid,  (field, ), (west_bc, ), (east_bc, ))
                KernelAbstractions.synchronize(backend)
                @test all((@views field[0   , 1:ny, 1:nz] .+ field[1, 1:ny, 1:nz]) ./ 2 .≈ west_bc.val)
                @test all( @views field[nx+1, 1:ny, 1:nz] .≈ east_bc.val)
            end
            @testset "y-dim" begin
                data(field) .= 0.0
                bc_array_south = KernelAbstractions.allocate(backend, Float64, (size(grid, 2), size(grid, 3)))
                bc_array_north = KernelAbstractions.allocate(backend, Float64, (size(grid, 2), size(grid, 3)))
                bc_array_south .= 1.0
                bc_array_north .= 0.5
                south_bc = DirichletBC{HalfCell}(bc_array_south)
                north_bc = DirichletBC{FullCell}(bc_array_north)
                discrete_bcs_y!(backend, 256, size(grid))(grid,  (field, ), (south_bc, ), (north_bc, ))
                KernelAbstractions.synchronize(backend)
                @test all((@views field[1:nx, 0   , 1:nz] .+ field[1:nx, 1, 1:nz]) ./ 2 .≈ south_bc.val)
                @test all( @views field[1:nx, ny+1, 1:nz] .≈ north_bc.val)
            end
            @testset "z-dim" begin
                data(field) .= 0.0
                bc_array_bot = KernelAbstractions.allocate(backend, Float64, (size(grid, 2), size(grid, 3)))
                bc_array_top = KernelAbstractions.allocate(backend, Float64, (size(grid, 2), size(grid, 3)))
                bc_array_bot .= 1.0
                bc_array_top .= 0.5
                bot_bc = DirichletBC{HalfCell}(bc_array_bot)
                top_bc = DirichletBC{FullCell}(bc_array_top)
                discrete_bcs_z!(backend, 256, size(grid))(grid,  (field, ), (bot_bc, ), (top_bc, ))
                KernelAbstractions.synchronize(backend)
                @test all((@views field[1:nx, 1:ny, 0   ] .+ field[1:nx, 1:ny, 2]) ./ 2 .≈ bot_bc.val)
                @test all( @views field[1:nx, 1:ny, nz+1] .≈ top_bc.val)
            end
        end
        @testset "no BC" begin
            @testset "x-dim" begin
                data(field) .= 0.0
                west_bc = NoBC()
                east_bc = NoBC()
                discrete_bcs_x!(backend, 256, size(grid))(grid, (field, ), (west_bc, ), (east_bc, ))
                KernelAbstractions.synchronize(backend)
                @test all(@views field[0   , 1:ny, 1:nz] .≈ 0.0)
                @test all(@views field[nx+1, 1:ny, 1:nz] .≈ 0.0)
            end
            @testset "y-dim" begin
                data(field) .= 0.0
                south_bc = NoBC()
                north_bc = NoBC()
                discrete_bcs_y!(backend, 256, size(grid))(grid, (field, ), (south_bc, ), (north_bc, ))
                KernelAbstractions.synchronize(backend)
                @test all(@views field[1:nx, 0   , 1:nz]  .≈ 0.0)
                @test all(@views field[1:nx, ny+1, 1:nz] .≈ 0.0)
            end
            @testset "z-dim" begin
                data(field) .= 0.0
                bot_bc = NoBC()
                top_bc = NoBC()
                discrete_bcs_z!(backend, 256, size(grid))(grid, (field, ), (bot_bc, ), (top_bc, ))
                KernelAbstractions.synchronize(backend)
                @test all(@views field[1:nx, 1:ny, 0   ] .≈ 0.0)
                @test all(@views field[1:nx, 1:ny, nz+1] .≈ 0.0)
            end
        end
    end
end
