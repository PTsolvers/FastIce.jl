function rotation_matrix(α,β,γ)
    sα,cα = sincos(α)
    sβ,cβ = sincos(β)
    sγ,cγ = sincos(γ)
    return Mat3(
        cα*cβ         ,sα*cβ         ,-sβ,
        cα*sβ*sγ-sα*cγ,sα*sβ*sγ+cα*cγ,cβ*sγ,
        cα*sβ*cγ+sα*sγ,sα*sβ*cγ-cα*sγ,cβ*cγ,
    )
end