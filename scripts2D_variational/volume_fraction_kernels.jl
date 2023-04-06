@tiny function _kernel_compute_volume_fractions_from_level_set!(wt, Ψ, dx, dy)
    ix, iy = @indices
    cell = Rect(Vec(0.0, 0.0), Vec(dx, dy))
    ω = GeometryBasics.volume(cell)
    @inline Ψ_ax(dix, diy) = 0.5 * (Ψ[ix+dix, iy+diy] + Ψ[ix+dix+1, iy+diy  ])
    @inline Ψ_ay(dix, diy) = 0.5 * (Ψ[ix+dix, iy+diy] + Ψ[ix+dix  , iy+diy+1])
    @inline Ψ_axy(dix, diy) = 0.25 * (Ψ[ix+dix, iy+diy  ] + Ψ[ix+dix+1, iy+diy  ] +
                                      Ψ[ix+dix, iy+diy+1] + Ψ[ix+dix+1, iy+diy+1])
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    # cell centers
    @inbounds if isin(wt.c)
        Ψs = Vec{4}(Ψ[ix, iy], Ψ[ix+1, iy], Ψ[ix+1, iy+1], Ψ[ix, iy+1])
        wt.c[ix, iy] = volfrac(cell, Ψs) / ω
    end
    # x faces
    @inbounds if isin(wt.x)
        Ψs = Vec{4}(Ψ_ax(0, 0), Ψ_ax(1, 0), Ψ_ax(1, 1), Ψ_ax(0, 1))
        wt.x[ix, iy] = volfrac(cell, Ψs) / ω
    end
    # y faces
    @inbounds if isin(wt.y)
        Ψs = Vec{4}(Ψ_ay(0, 0), Ψ_ay(1, 0), Ψ_ay(1, 1), Ψ_ay(0, 1))
        wt.y[ix, iy] = volfrac(cell, Ψs) / ω
    end
    # xy edges
    @inbounds if isin(wt.xy)
        Ψs = Vec{4}(Ψ_axy(0, 0), Ψ_axy(1, 0), Ψ_axy(1, 1), Ψ_axy(0, 1))
        wt.xy[ix, iy] = volfrac(cell, Ψs) / ω
    end
    return
end