@inline scalar_field(::Type{T}, nx, ny) where {T}  = field_array(T, nx, ny)
@inline vector_field(::Type{T}, nx, ny) where {T}  =  (x  = field_array(T, nx + 1, ny    ),
                                                       y  = field_array(T, nx    , ny + 1))
@inline tensor_field(::Type{T}, nx, ny) where {T}  = (xx  = field_array(T, nx    , ny    ),
                                                      yy  = field_array(T, nx    , ny    ),
                                                      xy  = field_array(T, nx - 1, ny - 1),
                                                      xyc = field_array(T, nx    , ny    ))
@inline volfrac_field(::Type{T}, nx, ny) where {T} = (c   = field_array(T, nx    , ny    ),
                                                      x   = field_array(T, nx + 1, ny    ),
                                                      y   = field_array(T, nx    , ny + 1),
                                                      xy  = field_array(T, nx - 1, ny - 1))

@tiny function _kernel_init!(Pr, τ, δτ, ε, V, ηs, ebg, ηs0, xv, yv)
    ix, iy = @indices()
    @inbounds if ix ∈ axes(Pr, 1) && iy ∈ axes(Pr, 2)
        Pr[ix, iy]     = 0.0
        τ.xx[ix, iy]   = 0.0
        τ.yy[ix, iy]   = 0.0
        τ.xyc[ix, iy]  = 0.0
        δτ.xx[ix, iy]  = 0.0
        δτ.yy[ix, iy]  = 0.0
        δτ.xyc[ix, iy] = 0.0
        ε.xx[ix, iy]   = 0.0
        ε.yy[ix, iy]   = 0.0
        ε.xyc[ix, iy]  = 0.0
        ηs[ix, iy]     = ηs0
    end
    @inbounds if ix ∈ axes(τ.xy, 1) && iy ∈ axes(τ.xy, 2)
        τ.xy[ix, iy]  = 0.0
        δτ.xy[ix, iy] = 0.0
        ε.xy[ix, iy]  = 0.0
    end
    if ix ∈ axes(V.x, 1) && iy ∈ axes(V.x, 2)
        @inbounds V.x[ix, iy] = -xv[ix] * ebg
    end
    if ix ∈ axes(V.y, 1) && iy ∈ axes(V.y, 2)
        @inbounds V.y[ix, iy] = yv[iy] * ebg
    end
end

@tiny function _kernel_update_vis_fields!(Vmag, Ψav, V, Ψ)
    ix, iy = @indices
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    @inbounds if isin(Ψ.not_air)
        pav = 0.0
        for idy = 1:2, idx = 1:2
            pav += Ψ.not_air[ix+idx, iy+idy]
        end
        Ψav.not_air[ix, iy] = pav / 8
    end
    # @inbounds if isin(Ψ.not_solid)
    #     pav = 0.0
    #     for idy = 1:2, idx = 1:2
    #         pav += Ψ.not_solid[ix+idx, iy+idy]
    #     end
    #     Ψav.not_solid[ix, iy] = pav / 8
    # end
    @inbounds if isin(Vmag)
        vxc = 0.5 * (V.x[ix+1, iy+1] + V.x[ix+2, iy+1])
        vyc = 0.5 * (V.y[ix+1, iy+1] + V.y[ix+1, iy+2])
        Vmag[ix, iy] = sqrt(vxc^2 + vyc^2)
    end
    return
end

const _init! = _kernel_init!(get_device())
const _update_vis! = _kernel_update_vis_fields!(get_device())

function init!(Pr, τ, δτ, ε, V, ηs, ebg, ηs0, xv, yv)
    wait(_init!(Pr, τ, δτ, ε, V, ηs, ebg, ηs0, xv, yv; ndrange=size(Pr) .+ 1))
    return
end

function update_vis!(Vmag, Ψav, V, Ψ)
    wait(_update_vis!(Vmag, Ψav, V, Ψ; ndrange=axes(Vmag)))
    return
end