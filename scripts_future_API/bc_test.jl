using FastIce.BoundaryConditions
using FastIce.Grids
using AMDGPU
using KernelAbstractions

function main(backend)
    grid    = CartesianGrid(origin = (0.0, 0.0, 0.0), extent = (1.0, 1.0, 1.0), size = (2, 2, 2))
    field   = KernelAbstractions.zeros(backend, Float64, size(grid) .+ 2)
    west_bc = DirichletBC{HalfCell}(1.0)
    east_bc = DirichletBC{FullCell}(0.5)

    discrete_bcs_x!(backend, 256, size(grid))(grid, (field, ), (west_bc, ), (east_bc, ))
    KernelAbstractions.synchronize(backend)

    @assert all((field[1, 2:end-1, 2:end-1] .+ field[2, 2:end-1, 2:end-1]) ./ 2 .≈ west_bc.val) 
    @assert all(field[end, 2:end-1, 2:end-1] .≈ east_bc.val) 

    return
end

main(ROCBackend())