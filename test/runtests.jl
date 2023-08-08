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
        grid    = CartesianGrid(origin = (0.0, 0.0, 0.0), extent = (1.0, 1.0, 1.0), size = (2, 2, 2))
        field   = KernelAbstractions.zeros(backend, Float64, size(grid) .+ 2)
        west_bc = DirichletBC{HalfCell}(1.0)
        east_bc = DirichletBC{FullCell}(0.5)

        discrete_bcs_x!(backend, 256, size(grid))(grid, (field, ), (west_bc, ), (east_bc, ))
        KernelAbstractions.synchronize(backend)

        @test all((field[1, 2:end-1, 2:end-1] .+ field[2, 2:end-1, 2:end-1]) ./ 2 .≈ west_bc.val) 
        @test all(field[end, 2:end-1, 2:end-1] .≈ east_bc.val) 
    end
end