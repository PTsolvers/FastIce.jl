module BoundaryConditions

using KernelAbstractions

struct West end
struct East end
struct North end
struct South end
struct Front end
struct Back end

@kernel function linear_bc_x!(A, ix1, ix2, a, b)
    iy, iz = @index(Global, NTuple)
    for i in eachindex(ix1)
        @inbounds A[ix1[i], iy+1, iz+1] = a[i] * A[ix2[i], iy+1, iz+1] + b[i]
    end
end

@kernel function linear_bc_y!(A, iy1, iy2, a, b)
    ix, iz = @index(Global, NTuple)
    for i in eachindex(iy1)
        @inbounds A[ix+1, iy1[i], iz+1] = a[i] * A[ix+1, iy2[i], iz+1] + b[i]
    end
end

@kernel function linear_bc_z!(A, iz1, iz2, a, b)
    ix, iy = @index(Global, NTuple)
    for i in eachindex(iy1)
        @inbounds A[ix+1, iy+1, iz1[i]] = a[i] * A[ix+1, iy+1, iz2[i]] + b[i]
    end
end

# 1. NoSlip
# 1.1 V.x[w] = Vp.x
# 1.2 V.y[w] = -V.y[e] + 2.0 * Vp.y
# 2. FreeSlip
# 2.1 V.x[w]  = Vp.x
# 2.2 τ.xy[w] = τp.xy
# 3. FreeSurface
# 3.1 τ.xy[w] = τp.xy
# 3.2 Pr[w] = - Pr[e] - 2.0 * σp
# 3.3 τ.xx[w] = -τ.xx[e]

function apply_velocity_bc!(::West, bc::DirichletBC)

    return
end

end