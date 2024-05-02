# 2D case
@kernel inbounds = true function _ω_from_ψ!(ω, ψ, g::StructuredGrid{2}, O=Offset())
    I = @index(Global, NTuple)
    i, j = I + O

    # cell center
    cell_cc = ((ψ[i, j], ψ[i+1, j]), (ψ[i, j+1], ψ[i+1, j+1]))
    ω.cc[i, j] = ψ2ω(cell_cc, Δ(g, Center(), i, j))

    # vertex
    cell_vv = ((lerp(ψ, Center(), g, i - 1, j - 1), lerp(ψ, Center(), g, i, j - 1)),
               (lerp(ψ, Center(), g, i - 1, j + 0), lerp(ψ, Center(), g, i, j + 0)))
    ω.vv[i, j] = ψ2ω(cell_vv, Δ(g, Vertex(), i, j))

    # locations
    loc_vc = (Vertex(), Center())
    loc_cv = (Center(), Vertex())

    # x interface
    cell_vc = ((lerp(ψ, loc_cv, g, i - 1, j + 0), lerp(ψ, loc_cv, g, i, j + 0)),
               (lerp(ψ, loc_cv, g, i - 1, j + 1), lerp(ψ, loc_cv, g, i, j + 1)))
    ω.vc[i, j] = ψ2ω(cell_vc, Δ(g, loc_vc, i, j))

    # y interface
    cell_cv = ((lerp(ψ, loc_vc, g, i, j - 1), lerp(ψ, loc_vc, g, i + 1, j - 1)),
               (lerp(ψ, loc_vc, g, i, j + 0), lerp(ψ, loc_vc, g, i + 1, j + 0)))
    ω.cv[i, j] = ψ2ω(cell_cv, Δ(g, loc_cv, i, j))
end

# 3D case
@kernel inbounds = true function _ω_from_ψ!(ω, ψ, g::StructuredGrid{3}, O=Offset())
    I = @index(Global, NTuple)
    i, j, k = I + O

    # cell center
    cell_ccc = (((ψ[i, j, k+0], ψ[i+1, j, k+0]), (ψ[i, j+1, k+0], ψ[i+1, j+1, k+0])),
                ((ψ[i, j, k+1], ψ[i+1, j, k+1]), (ψ[i, j+1, k+1], ψ[i+1, j+1, k+1])))
    ω.ccc[i, j, k] = ψ2ω(cell_ccc, Δ(g, Center(), i, j, k))

    # vertex
    cell_vvv = (((lerp(ψ, Center(), g, i - 1, j - 1, k - 1), lerp(ψ, Center(), g, i, j - 1, k - 1)),
                 (lerp(ψ, Center(), g, i - 1, j + 0, k - 1), lerp(ψ, Center(), g, i, j + 0, k - 1))),
                ((lerp(ψ, Center(), g, i - 1, j - 1, k + 0), lerp(ψ, Center(), g, i, j - 1, k + 0)),
                 (lerp(ψ, Center(), g, i - 1, j + 0, k + 0), lerp(ψ, Center(), g, i, j + 0, k + 0))))
    ω.vvv[i, j, k] = ψ2ω(cell_vvv, Δ(g, Vertex(), i, j, k))

    # locations
    loc_vcc = (Vertex(), Center(), Center())
    loc_cvc = (Center(), Vertex(), Center())
    loc_ccv = (Center(), Center(), Vertex())

    loc_vvc = (Vertex(), Vertex(), Center())
    loc_vcv = (Vertex(), Center(), Vertex())
    loc_cvv = (Center(), Vertex(), Vertex())

    # x interface
    cell_vcc = (((lerp(ψ, loc_cvv, g, i - 1, j + 0, k + 0), lerp(ψ, loc_cvv, g, i, j + 0, k + 0)),
                 (lerp(ψ, loc_cvv, g, i - 1, j + 1, k + 0), lerp(ψ, loc_cvv, g, i, j + 1, k + 0))),
                ((lerp(ψ, loc_cvv, g, i - 1, j + 0, k + 1), lerp(ψ, loc_cvv, g, i, j + 0, k + 1)),
                 (lerp(ψ, loc_cvv, g, i - 1, j + 1, k + 1), lerp(ψ, loc_cvv, g, i, j + 1, k + 1))))
    ω.vcc[i, j, k] = ψ2ω(cell_vcc, Δ(g, loc_vcc, i, j, k))

    # y interface
    cell_cvc = (((lerp(ψ, loc_vcv, g, i, j - 1, k + 0), lerp(ψ, loc_vcv, g, i + 1, j - 1, k + 0)),
                 (lerp(ψ, loc_vcv, g, i, j + 0, k + 0), lerp(ψ, loc_vcv, g, i + 1, j + 0, k + 0))),
                ((lerp(ψ, loc_vcv, g, i, j - 1, k + 1), lerp(ψ, loc_vcv, g, i + 1, j - 1, k + 1)),
                 (lerp(ψ, loc_vcv, g, i, j + 0, k + 1), lerp(ψ, loc_vcv, g, i + 1, j + 0, k + 1))))
    ω.cvc[i, j, k] = ψ2ω(cell_cvc, Δ(g, loc_cvc, i, j, k))

    # z interface
    cell_ccv = (((lerp(ψ, loc_vvc, g, i, j + 0, k - 1), lerp(ψ, loc_vvc, g, i + 1, j + 0, k - 1)),
                 (lerp(ψ, loc_vvc, g, i, j + 1, k - 1), lerp(ψ, loc_vvc, g, i + 1, j + 1, k - 1))),
                ((lerp(ψ, loc_vvc, g, i, j + 0, k + 0), lerp(ψ, loc_vvc, g, i + 1, j + 0, k + 0)),
                 (lerp(ψ, loc_vvc, g, i, j + 1, k + 0), lerp(ψ, loc_vvc, g, i + 1, j + 1, k + 0))))
    ω.ccv[i, j, k] = ψ2ω(cell_ccv, Δ(g, loc_ccv, i, j, k))

    # xy edge
    cell_vvc = (((lerp(ψ, loc_ccv, g, i - 1, j - 1, k + 0), lerp(ψ, loc_ccv, g, i, j - 1, k + 0)),
                 (lerp(ψ, loc_ccv, g, i - 1, j + 0, k + 0), lerp(ψ, loc_ccv, g, i, j + 0, k + 0))),
                ((lerp(ψ, loc_ccv, g, i - 1, j - 1, k + 1), lerp(ψ, loc_ccv, g, i, j - 1, k + 1)),
                 (lerp(ψ, loc_ccv, g, i - 1, j + 0, k + 1), lerp(ψ, loc_ccv, g, i, j + 0, k + 1))))
    ω.vvc[i, j, k] = ψ2ω(cell_vvc, Δ(g, loc_vvc, i, j, k))

    # xz edge
    cell_vcv = (((lerp(ψ, loc_cvc, g, i - 1, j + 0, k - 1), lerp(ψ, loc_cvc, g, i, j + 0, k - 1)),
                 (lerp(ψ, loc_cvc, g, i - 1, j + 1, k - 1), lerp(ψ, loc_cvc, g, i, j + 1, k - 1))),
                ((lerp(ψ, loc_cvc, g, i - 1, j + 0, k + 0), lerp(ψ, loc_cvc, g, i, j + 0, k + 0)),
                 (lerp(ψ, loc_cvc, g, i - 1, j + 1, k + 0), lerp(ψ, loc_cvc, g, i, j + 1, k + 0))))
    ω.vcv[i, j, k] = ψ2ω(cell_vcv, Δ(g, loc_vcv, i, j, k))

    # yz edge
    cell_cvv = (((lerp(ψ, loc_vcc, g, i, j - 1, k - 1), lerp(ψ, loc_vcc, g, i + 1, j - 1, k - 1)),
                 (lerp(ψ, loc_vcc, g, i, j + 0, k - 1), lerp(ψ, loc_vcc, g, i + 1, j + 0, k - 1))),
                ((lerp(ψ, loc_vcc, g, i, j - 1, k + 0), lerp(ψ, loc_vcc, g, i + 1, j - 1, k + 0)),
                 (lerp(ψ, loc_vcc, g, i, j + 0, k + 0), lerp(ψ, loc_vcc, g, i + 1, j + 0, k + 0))))
    ω.cvv[i, j, k] = ψ2ω(cell_cvv, Δ(g, loc_cvv, i, j, k))
end

const AnyFieldMask = Union{FieldMask1D, FieldMask2D, FieldMask3D}

function ω_from_ψ!(arch::Architecture, launch::Launcher, ω::AnyFieldMask, ψ::AbstractField, grid::StructuredGrid)
    launch(arch, grid, _ω_from_ψ! => (ω, ψ, grid))
    return
end
