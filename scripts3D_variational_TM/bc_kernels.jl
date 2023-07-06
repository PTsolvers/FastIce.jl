@tiny function _kernel_bc_x_dirichlet!(val,arrays...)
    iy,iz = @indices
    for A in arrays
        if iy ∈ axes(A,2) && iz ∈ axes(A,3)
            @inbounds A[1  ,iy,iz] = val
            @inbounds A[end,iy,iz] = val
        end
    end
    return
end

@tiny function _kernel_bc_y_dirichlet!(val, arrays...)
    ix,iz = @indices
    for A in arrays
        if ix ∈ axes(A,1) && iz ∈ axes(A,3)
            @inbounds A[ix,1  ,iz] = val
            @inbounds A[ix,end,iz] = val
        end
    end
    return
end

@tiny function _kernel_bc_z_dirichlet!(val, arrays...)
    ix,iy = @indices
    for A in arrays
        if ix ∈ axes(A,1) && iy ∈ axes(A,2)
            @inbounds A[ix,iy,1  ] = val
            @inbounds A[ix,iy,end] = val
        end
    end
    return
end

@tiny function _kernel_bc_x_neumann!(val, arrays...)
    iy,iz = @indices
    for A in arrays
        if iy ∈ axes(A,2) && iz ∈ axes(A,3)
            @inbounds A[1  ,iy,iz] = A[2    ,iy,iz] + val
            @inbounds A[end,iy,iz] = A[end-1,iy,iz] + val
        end
    end
    return
end

@tiny function _kernel_bc_y_neumann!(val, arrays...)
    ix,iz = @indices
    for A in arrays
        if ix ∈ axes(A,1) && iz ∈ axes(A,3)
            @inbounds A[ix,1  ,iz] = A[ix,2    ,iz] + val
            @inbounds A[ix,end,iz] = A[ix,end-1,iz] + val
        end
    end
    return
end

@tiny function _kernel_bc_z_neumann!(val, arrays...)
    ix,iy = @indices
    for A in arrays
        if ix ∈ axes(A,1) && iy ∈ axes(A,2)
            @inbounds A[ix,iy,1  ] = A[ix,iy,2    ] + val
            @inbounds A[ix,iy,end] = A[ix,iy,end-1] + val
        end
    end
    return
end