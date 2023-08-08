using Test

using FastIce.Grids
using FastIce.BoundaryConditions
using KernelAbstractions

# add KA backends
backends = KernelAbstractions.Backend[CPU(), ]

using CUDA
CUDA.functional() && push!(backends, CUDABackend())

using AMDGPU
AMDGPU.functional() && push!(backends, ROCBackend())

@testset "backend $backend" for backend in backends
    @testset "boundary conditions" begin
        grid  = CartesianGrid(origin = (0.0, 0.0, 0.0), extent = (1.0, 1.0, 1.0), size = (2, 2, 2))
        field = KernelAbstractions.zeros(backend, Float64, size(grid) .+ 2)
        @testset "value" begin
            @testset "x-dim" begin
                field .= 0.0
                west_bc = DirichletBC{HalfCell}(1.0)
                east_bc = DirichletBC{FullCell}(0.5)
                discrete_bcs_x!(backend, 256, size(grid))(grid, (@view(field[:, 2:end-1, 2:end-1]), ), (west_bc, ), (east_bc, ))
                KernelAbstractions.synchronize(backend)
                @test all((field[1, 2:end-1, 2:end-1] .+ field[2, 2:end-1, 2:end-1]) ./ 2 .≈ west_bc.val)
                @test all(field[end, 2:end-1, 2:end-1] .≈ east_bc.val)
            end
            @testset "y-dim" begin
                field .= 0.0
                south_bc = DirichletBC{HalfCell}(1.0)
                north_bc = DirichletBC{FullCell}(0.5)
                discrete_bcs_y!(backend, 256, size(grid))(grid, (@view(field[2:end-1, :, 2:end-1]), ), (south_bc, ), (north_bc, ))
                KernelAbstractions.synchronize(backend)
                @test all((field[2:end-1, 1, 2:end-1] .+ field[2:end-1, 2, 2:end-1]) ./ 2 .≈ south_bc.val)
                @test all(field[2:end-1, end, 2:end-1] .≈ north_bc.val)
            end
            @testset "z-dim" begin
                field .= 0.0
                bot_bc = DirichletBC{HalfCell}(1.0)
                top_bc = DirichletBC{FullCell}(0.5)
                discrete_bcs_z!(backend, 256, size(grid))(grid, (@view(field[2:end-1, 2:end-1, :]), ), (bot_bc, ), (top_bc, ))
                KernelAbstractions.synchronize(backend)
                @test all((field[2:end-1, 2:end-1, 1] .+ field[2:end-1, 2:end-1, 2]) ./ 2 .≈ bot_bc.val)
                @test all(field[2:end-1, 2:end-1, end] .≈ top_bc.val)
            end
        end
        @testset "array" begin
            @testset "x-dim" begin
                field .= 0.0
                bc_array_west = KernelAbstractions.allocate(backend, Float64, (size(grid, 2), size(grid, 3)))
                bc_array_east = KernelAbstractions.allocate(backend, Float64, (size(grid, 2), size(grid, 3)))
                bc_array_west .= 1.0
                bc_array_east .= 0.5
                west_bc = DirichletBC{HalfCell}(bc_array_west)
                east_bc = DirichletBC{FullCell}(bc_array_east)
                discrete_bcs_x!(backend, 256, size(grid))(grid,  (@view(field[:, 2:end-1, 2:end-1]), ), (west_bc, ), (east_bc, ))
                KernelAbstractions.synchronize(backend)
                @test all((field[1, 2:end-1, 2:end-1] .+ field[2, 2:end-1, 2:end-1]) ./ 2 .≈ west_bc.val)
                @test all(field[end, 2:end-1, 2:end-1] .≈ east_bc.val)
            end
            @testset "y-dim" begin
                field .= 0.0
                bc_array_south = KernelAbstractions.allocate(backend, Float64, (size(grid, 2), size(grid, 3)))
                bc_array_north = KernelAbstractions.allocate(backend, Float64, (size(grid, 2), size(grid, 3)))
                bc_array_south .= 1.0
                bc_array_north .= 0.5
                south_bc = DirichletBC{HalfCell}(bc_array_south)
                north_bc = DirichletBC{FullCell}(bc_array_north)
                discrete_bcs_y!(backend, 256, size(grid))(grid,  (@view(field[2:end-1, :, 2:end-1]), ), (south_bc, ), (north_bc, ))
                KernelAbstractions.synchronize(backend)
                @test all((field[2:end-1, 1, 2:end-1] .+ field[2:end-1, 2, 2:end-1]) ./ 2 .≈ south_bc.val)
                @test all(field[2:end-1, end, 2:end-1] .≈ north_bc.val)
            end
            @testset "z-dim" begin
                field .= 0.0
                bc_array_bot = KernelAbstractions.allocate(backend, Float64, (size(grid, 2), size(grid, 3)))
                bc_array_top = KernelAbstractions.allocate(backend, Float64, (size(grid, 2), size(grid, 3)))
                bc_array_bot .= 1.0
                bc_array_top .= 0.5
                bot_bc = DirichletBC{HalfCell}(bc_array_bot)
                top_bc = DirichletBC{FullCell}(bc_array_top)
                discrete_bcs_z!(backend, 256, size(grid))(grid,  (@view(field[2:end-1, 2:end-1, :]), ), (bot_bc, ), (top_bc, ))
                KernelAbstractions.synchronize(backend)
                @test all((field[2:end-1, 2:end-1, 1] .+ field[2:end-1, 2:end-1, 2]) ./ 2 .≈ bot_bc.val)
                @test all(field[2:end-1, 2:end-1, end] .≈ top_bc.val)
            end
        end
    end
end