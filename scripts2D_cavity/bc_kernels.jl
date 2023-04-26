@tiny function _kernel_bc_x_dirichlet!(val, arrays...)
    iy, = @indices
    for A in arrays
        if iy ∈ axes(A, 2)
            @inbounds A[1  ,iy] = val
            @inbounds A[end,iy] = val
        end
    end
    return
end

@tiny function _kernel_bc_x_dirichlet!(vec::NTuple{2}, arrays...)
    iy, = @indices
    for A in arrays
        if iy ∈ axes(A, 2)
            @inbounds A[1  ,iy] = vec[1]
            @inbounds A[end,iy] = vec[2]
        end
    end
    return
end

@tiny function _kernel_bc_y_dirichlet!(val, arrays...)
    ix, = @indices
    for A in arrays
        if ix ∈ axes(A, 1)
            @inbounds A[ix,1  ] = val
            @inbounds A[ix,end] = val
        end
    end
    return
end

@tiny function _kernel_bc_y_dirichlet!(vec::NTuple{2}, arrays...)
    ix, = @indices
    for A in arrays
        if ix ∈ axes(A, 1)
            @inbounds A[ix,1  ] = vec[1]
            @inbounds A[ix,end] = vec[2]
        end
    end
    return
end

@tiny function _kernel_bc_x_neumann!(val, arrays...)
    iy, = @indices
    for A in arrays
        if iy ∈ axes(A, 2)
            @inbounds A[1, iy]  = A[2   , iy] + val
            @inbounds A[end,iy] = A[end-1,iy] + val
        end
    end
    return
end

@tiny function _kernel_bc_y_neumann!(val, arrays...)
    ix, = @indices
    for A in arrays
        if ix ∈ axes(A, 1)
            @inbounds A[ix,1  ] = A[ix,2    ] + val
            @inbounds A[ix,end] = A[ix,end-1] + val
        end
    end
    return
end