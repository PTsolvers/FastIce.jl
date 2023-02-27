using Plots, Plots.Measures
using LaTeXStrings

default(fontfamily="Computer Modern", linewidth=4,  markershape=:circle, markersize=4,
        framestyle=:box, fillalpha=0.4, margin=5mm)
scalefontsizes(); scalefontsizes(1.3)

nz    = (63, 127, 255, 511)
iters = (160,320,640,1152)

T_init  = (0.294776409,0.605874031,3.123633533,43.06257976)

png(plot(collect(nz),collect(1e9.*T_init./(nz.^3)./iters),
         xticks=(nz, string.(nz).*L"^3"), legend=false,
         xlabel="Number of grid points", ylabel="Initialisation time per giga-point [s]",yscale=:log10,
         dpi=150,size=(600, 380)),"init_strong_scale_lumi.png")
