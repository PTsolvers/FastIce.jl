Base.@propagate_inbounds @generated function within(grid::CartesianGrid{N}, f::Field{T,N}, I::CartesianIndex{N}) where {T,N}
    quote
        Base.Cartesian.@nall $N i->I[i] <= size(grid, location(f, Val(i)), i)
    end
end

"Update viscosity using relaxation in log-space. Improves stability of iterative methods"
@kernel function update_η!(η, η_rh, χ, fields, grid, offset=nothing)
    I = @index(Global, Cartesian)
    isnothing(offset) || (I += offset)
    @inbounds begin
        ηt = η_rh(grid, I, fields)
        η[I] = exp(log(η[I]) * (1 - χ) + log(ηt) * χ)
    end
end
