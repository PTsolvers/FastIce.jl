include("geometry.jl")

# @tiny function _kernel_compute_levelset_from_polygon!(ψ, xv, yv, mc)
#     ix, iy = @indices()
#     @inbounds ψ[ix, iy] = signed_distance(Point(xv[ix], yv[iy]), mc)
# end

@tiny function _kernel_compute_levelset_from_polygon!(op, ψ, xv, yv, mc)
    ix, iy = @indices()
    @inbounds ψ[ix, iy] = op(ψ[ix, iy], signed_distance(Point(xv[ix], yv[iy]), mc))
end