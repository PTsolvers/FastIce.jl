struct ZeroField{T} <: AbstractField{T,1,Nothing} end

@inline Base.getindex(::ZeroField{T}, inds...) where {T} = zero(T)
@inline Base.size(::ZeroField) = (1,)

struct OneField{T} <: AbstractField{T,1,Nothing} end

@inline Base.getindex(::OneField{T}, inds...) where {T} = one(T)
@inline Base.size(::OneField) = (1,)

struct ConstantField{T} <: AbstractField{T,1,Nothing}
    value::T
end

@inline Base.getindex(f::ConstantField, inds...) = f.value
@inline Base.size(::ConstantField) = (1,)
