include("common.jl")

using FastIce.Grids
using FastIce.Fields
using FastIce.BoundaryConditions

@testset "backend $backend" for backend in backends
    @testset "boundary conditions" begin
        nx, ny, nz = 2, 2, 2
        grid = CartesianGrid(; origin=(0.0, 0.0, 0.0), extent=(1.0, 1.0, 1.0), size=(nx, ny, nz))
        field = Field(backend, grid, Center(); halo=1)
        @testset "value" begin
            @testset "x-dim" begin
                data(field) .= 0.0
                west_bc = FieldBoundaryConditions((field,), (DirichletBC{HalfCell}(1.0),))
                east_bc = FieldBoundaryConditions((field,), (DirichletBC{FullCell}(0.5),))
                apply_boundary_conditions!(Val(1), Val(1), backend, grid, west_bc; async=false)
                apply_boundary_conditions!(Val(2), Val(1), backend, grid, east_bc; async=false)
                @test all((parent(field)[1, 2:ny+1, 2:nz+1] .+ parent(field)[2, 2:ny+1, 2:nz+1]) ./ 2 .≈ west_bc.conditions[1].condition)
                @test all(parent(field)[nx+2, 2:ny+1, 2:nz+1] .≈ east_bc.conditions[1].condition)
            end
            @testset "y-dim" begin
                data(field) .= 0.0
                south_bc = FieldBoundaryConditions((field,), (DirichletBC{HalfCell}(1.0),))
                north_bc = FieldBoundaryConditions((field,), (DirichletBC{FullCell}(0.5),))
                apply_boundary_conditions!(Val(1), Val(2), backend, grid, south_bc; async=false)
                apply_boundary_conditions!(Val(2), Val(2), backend, grid, north_bc; async=false)
                @test all((parent(field)[2:nx+1, 1, 2:nz+1] .+ parent(field)[2:nx+1, 2, 2:nz+1]) ./ 2 .≈ south_bc.conditions[1].condition)
                @test all(parent(field)[2:nx+1, ny+2, 2:nz+1] .≈ north_bc.conditions[1].condition)
            end
            @testset "z-dim" begin
                data(field) .= 0.0
                bottom_bc = FieldBoundaryConditions((field,), (DirichletBC{HalfCell}(1.0),))
                top_bc = FieldBoundaryConditions((field,), (DirichletBC{FullCell}(0.5),))
                apply_boundary_conditions!(Val(1), Val(3), backend, grid, bottom_bc; async=false)
                apply_boundary_conditions!(Val(2), Val(3), backend, grid, top_bc; async=false)
                @test all((parent(field)[2:nx+1, 2:ny+1, 1] .+ parent(field)[2:nx+1, 2:ny+1, 2]) ./ 2 .≈ bottom_bc.conditions[1].condition)
                @test all(parent(field)[2:nx+1, 2:ny+1, nz+2] .≈ top_bc.conditions[1].condition)
            end
        end
        @testset "array" begin
            @testset "x-dim" begin
                data(field) .= 0.0
                bc_array_west = KernelAbstractions.allocate(backend, Float64, (size(grid, 2), size(grid, 3)))
                bc_array_east = KernelAbstractions.allocate(backend, Float64, (size(grid, 2), size(grid, 3)))
                bc_array_west .= 1.0
                bc_array_east .= 0.5
                west_bc = FieldBoundaryConditions((field,), (DirichletBC{HalfCell}(bc_array_west),))
                east_bc = FieldBoundaryConditions((field,), (DirichletBC{FullCell}(bc_array_east),))
                apply_boundary_conditions!(Val(1), Val(1), backend, grid, west_bc; async=false)
                apply_boundary_conditions!(Val(2), Val(1), backend, grid, east_bc; async=false)
                @test all((parent(field)[1, 2:ny+1, 2:nz+1] .+ parent(field)[2, 2:ny+1, 2:nz+1]) ./ 2 .≈ west_bc.conditions[1].condition)
                @test all(parent(field)[nx+2, 2:ny+1, 2:nz+1] .≈ east_bc.conditions[1].condition)
            end
            @testset "y-dim" begin
                data(field) .= 0.0
                bc_array_south = KernelAbstractions.allocate(backend, Float64, (size(grid, 2), size(grid, 3)))
                bc_array_north = KernelAbstractions.allocate(backend, Float64, (size(grid, 2), size(grid, 3)))
                bc_array_south .= 1.0
                bc_array_north .= 0.5
                south_bc = FieldBoundaryConditions((field,), (DirichletBC{HalfCell}(bc_array_south),))
                north_bc = FieldBoundaryConditions((field,), (DirichletBC{FullCell}(bc_array_north),))
                apply_boundary_conditions!(Val(1), Val(2), backend, grid, south_bc; async=false)
                apply_boundary_conditions!(Val(2), Val(2), backend, grid, north_bc; async=false)
                @test all((parent(field)[2:nx+1, 1, 2:nz+1] .+ parent(field)[2:nx+1, 2, 2:nz+1]) ./ 2 .≈ south_bc.conditions[1].condition)
                @test all(parent(field)[2:nx+1, ny+2, 2:nz+1] .≈ north_bc.conditions[1].condition)
            end
            @testset "z-dim" begin
                data(field) .= 0.0
                bc_array_bot = KernelAbstractions.allocate(backend, Float64, (size(grid, 2), size(grid, 3)))
                bc_array_top = KernelAbstractions.allocate(backend, Float64, (size(grid, 2), size(grid, 3)))
                bc_array_bot .= 1.0
                bc_array_top .= 0.5
                bottom_bc = FieldBoundaryConditions((field,), (DirichletBC{HalfCell}(bc_array_bot),))
                top_bc = FieldBoundaryConditions((field,), (DirichletBC{FullCell}(bc_array_top),))
                apply_boundary_conditions!(Val(1), Val(3), backend, grid, bottom_bc; async=false)
                apply_boundary_conditions!(Val(2), Val(3), backend, grid, top_bc; async=false)
                @test all((parent(field)[2:nx+1, 2:ny+1, 1] .+ parent(field)[2:nx+1, 2:ny+1, 2]) ./ 2 .≈ bottom_bc.conditions[1].condition)
                @test all(parent(field)[2:nx+1, 2:ny+1, nz+2] .≈ top_bc.conditions[1].condition)
            end
        end
    end
end
