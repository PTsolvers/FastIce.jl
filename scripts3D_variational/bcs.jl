include("bc_kernels.jl")

const _bc_x_dirichlet! = Kernel(_kernel_bc_x_dirichlet!, get_device())
const _bc_y_dirichlet! = Kernel(_kernel_bc_y_dirichlet!, get_device())
const _bc_z_dirichlet! = Kernel(_kernel_bc_z_dirichlet!, get_device())

const _bc_x_neumann! = Kernel(_kernel_bc_x_neumann!, get_device())
const _bc_y_neumann! = Kernel(_kernel_bc_y_neumann!, get_device())
const _bc_z_neumann! = Kernel(_kernel_bc_z_neumann!, get_device())

for fname in (:bx_x_dirichlet!,:bc_x_neumann!)
    @eval begin
        function $fname(val,arrays...)
            ax = axes(arrays[1])[[2,3]]
            for A in arrays[2:end]
                ax = union.(ax,axes(A)[[2,3]])
            end
            wait($(Symbol(:_,fname))(val,arrays...;ndrange=ax))
            return
        end
    end
end

for fname in (:bx_y_dirichlet!,:bc_y_neumann!)
    @eval begin
        function $fname(val,arrays...)
            ax = axes(arrays[1])[[1,3]]
            for A in arrays[2:end]
                ax = union.(ax,axes(A)[[1,3]])
            end
            wait($(Symbol(:_,fname))(val,arrays...;ndrange=ax))
            return
        end
    end
end

for fname in (:bx_z_dirichlet!,:bc_z_neumann!)
    @eval begin
        function $fname(val,arrays...)
            ax = axes(arrays[1])[[1,2]]
            for A in arrays[2:end]
                ax = union.(ax,axes(A)[[1,2]])
            end
            wait($(Symbol(:_,fname))(val,arrays...;ndrange=ax))
            return
        end
    end
end