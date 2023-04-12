include("geometry.jl")

@tiny function _kernel_compute_levelset_from_polygon!(op, ψ, xv, yv, mc)
    ix, iy = @indices
    @inbounds ψ[ix, iy] = op(ψ[ix, iy], signed_distance(Point(xv[ix], yv[iy]), mc))
end

@tiny function _kernel_extrapolate_with_levelset!(∂A_∂τ,A,Ψ,Δx,Δy)
    @inline S(x) = x/sqrt(x^2+max(Δx,Δy)^2)
    @inline Ψ_c(_ix,_iy) = 0.25*(Ψ[_ix,_iy] + Ψ[_ix+1,_iy] + Ψ[_ix,_iy+1] + Ψ[_ix+1,_iy+1])
    @inline Ψ_x(_ix,_iy) = 0.5*(Ψ[_ix,_iy] + Ψ[_ix+1,_iy])
    @inline Ψ_y(_ix,_iy) = 0.5*(Ψ[_ix,_iy] + Ψ[_ix,_iy+1])
    ix,iy = @indices
    if Ψ[ix,iy] > 0 && Ψ[ix+1,iy] > 0 && Ψ[ix,iy+1] > 0 && Ψ[ix+1,iy+1] > 0
        s     = S(Ψ_c(ix,iy))
        ∇Ψx   = (Ψ_y(ix+1,iy) - Ψ_y(ix,iy))/Δx
        ∇Ψy   = (Ψ_x(ix,iy+1) - Ψ_x(ix,iy))/Δy
        nx,ny = ∇Ψx/sqrt(∇Ψx^2 + ∇Ψy^2), ∇Ψy/sqrt(∇Ψx^2 + ∇Ψy^2)
        Fx    = max(s*nx,0)*(A[ix,iy]-A[ix-1,iy])/Δx + min(s*nx,0)*(A[ix+1,iy]-A[ix,iy])/Δx
        Fy    = max(s*ny,0)*(A[ix,iy]-A[ix,iy-1])/Δy + min(s*ny,0)*(A[ix,iy+1]-A[ix,iy])/Δy
        ∂A_∂τ[ix-1,iy-1] = -(Fx + Fy)
    end
end