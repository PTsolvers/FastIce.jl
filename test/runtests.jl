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
                discrete_bcs_x!(backend, 256, size(grid))(grid, (field, ), (west_bc, ), (east_bc, ))
                KernelAbstractions.synchronize(backend)
                @test all((field[1, 2:end-1, 2:end-1] .+ field[2, 2:end-1, 2:end-1]) ./ 2 .≈ west_bc.val)
                @test all(field[end, 2:end-1, 2:end-1] .≈ east_bc.val)
            end
            @testset "y-dim" begin
                field .= 0.0
                south_bc = DirichletBC{HalfCell}(1.0)
                north_bc = DirichletBC{FullCell}(0.5)
                discrete_bcs_y!(backend, 256, size(grid))(grid, (field, ), (south_bc, ), (north_bc, ))
                KernelAbstractions.synchronize(backend)
                @test all((field[2:end-1, 1, 2:end-1] .+ field[2:end-1, 2, 2:end-1]) ./ 2 .≈ south_bc.val)
                @test all(field[2:end-1, end, 2:end-1] .≈ north_bc.val)
            end
            @testset "z-dim" begin
                field .= 0.0
                bot_bc = DirichletBC{HalfCell}(1.0)
                top_bc = DirichletBC{FullCell}(0.5)
                discrete_bcs_z!(backend, 256, size(grid))(grid, (field, ), (bot_bc, ), (top_bc, ))
                KernelAbstractions.synchronize(backend)
                @test all((field[2:end-1, 2:end-1, 1] .+ field[2:end-1, 2:end-1, 2]) ./ 2 .≈ bot_bc.val)
                @test all(field[2:end-1, 2:end-1, end] .≈ top_bc.val)
            end
        end
        # @testset "array" begin
        #     @testset "x-dim" begin
        #         field .= 0.0
        #         west_bc = DirichletBC{HalfCell}(1.0)
        #         east_bc = DirichletBC{FullCell}(0.5)
        #         discrete_bcs_x!(backend, 256, size(grid))(grid, (field, ), (west_bc, ), (east_bc, ))
        #         KernelAbstractions.synchronize(backend)
        #         @test all((field[1, 2:end-1, 2:end-1] .+ field[2, 2:end-1, 2:end-1]) ./ 2 .≈ west_bc.val)
        #         @test all(field[end, 2:end-1, 2:end-1] .≈ east_bc.val)
        #     end
        #     @testset "y-dim" begin
        #         field .= 0.0
        #         south_bc = DirichletBC{HalfCell}(1.0)
        #         north_bc = DirichletBC{FullCell}(0.5)
        #         discrete_bcs_y!(backend, 256, size(grid))(grid, (field, ), (south_bc, ), (north_bc, ))
        #         KernelAbstractions.synchronize(backend)
        #         @test all((field[2:end-1, 1, 2:end-1] .+ field[2:end-1, 2, 2:end-1]) ./ 2 .≈ south_bc.val)
        #         @test all(field[2:end-1, end, 2:end-1] .≈ north_bc.val)
        #     end
        #     @testset "z-dim" begin
        #         field .= 0.0
        #         bot_bc = DirichletBC{HalfCell}(1.0)
        #         top_bc = DirichletBC{FullCell}(0.5)
        #         discrete_bcs_z!(backend, 256, size(grid))(grid, (field, ), (bot_bc, ), (top_bc, ))
        #         KernelAbstractions.synchronize(backend)
        #         @test all((field[2:end-1, 2:end-1, 1] .+ field[2:end-1, 2:end-1, 2]) ./ 2 .≈ bot_bc.val)
        #         @test all(field[2:end-1, 2:end-1, end] .≈ top_bc.val)
        #     end
        # end
    end
end