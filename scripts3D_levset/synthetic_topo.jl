using HDF5

function generate_synthetic_topo(nx,ny)
    lx,ly = 100.0,100.0
    ox,oy = 0.0,0.0
    dx,dy = lx/nx,ly/ny
    xc,yc = LinRange(dx/2,lx-dx/2,nx),LinRange(dy/2,ly-dy/2,ny)
    bed   = @. (5*(xc-0.5lx)/lx + 5*(yc'-0.5ly)/ly + 4 + 5*(0.5*(1 + sin(4π*xc/lx).*cos(4π*yc'/ly))))
    ice   = @. -1 + 11*sqrt(max(0.0,(1 - ((xc-0.5lx)/(0.45lx))^2 - ((yc'-0.5ly)/(0.45ly))^2)))
    mask  = ones(size(ice)); #@. mask[ice==0.0] = NaN
    h5open("data/synthetic_topo/dem_$(nx)_$(ny).h5","w") do io
        @write io bed; @write io ice
        @write io lx; @write io ly
        @write io ox; @write io oy
    end
    fig = Figure(fontsize=32,resolution=(2000,1600))
    ax  = Axis3(fig[1,1];aspect=:data)
    surface!(ax,xc,yc,bed;colormap=:turbo)
    surface!(ax,xc,yc,bed.+mask.*ice;colormap=:winter)
    display(fig)
    return
end