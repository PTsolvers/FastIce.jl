@tiny function _kernel_compute_volume_fractions_from_level_set!(wt,Ψ,dx,dy,dz)
    ix,iy,iz = @indices
    cell = Rect(Vec(0.0,0.0,0.0), Vec(dx,dy,dz))
    ω = GeometryBasics.volume(cell)
    @inline Ψ_ax(dix,diy,diz) = 0.5*(Ψ[ix+dix,iy+diy,iz+diz]+Ψ[ix+dix+1,iy+diy,iz+diz])
    @inline Ψ_ay(dix,diy,diz) = 0.5*(Ψ[ix+dix,iy+diy,iz+diz]+Ψ[ix+dix,iy+diy+1,iz+diz])
    @inline Ψ_az(dix,diy,diz) = 0.5*(Ψ[ix+dix,iy+diy,iz+diz]+Ψ[ix+dix,iy+diy,iz+diz+1])
    @inline Ψ_axy(dix,diy,diz) = 0.25*(Ψ[ix+dix  ,iy+diy  ,iz+diz+1]+Ψ[ix+dix+1,iy+diy  ,iz+diz+1]+
                                       Ψ[ix+dix  ,iy+diy+1,iz+diz+1]+Ψ[ix+dix+1,iy+diy+1,iz+diz+1])
    @inline Ψ_axz(dix,diy,diz) = 0.25*(Ψ[ix+dix  ,iy+diy+1,iz+diz  ]+Ψ[ix+dix+1,iy+diy+1,iz+diz  ]+
                                       Ψ[ix+dix  ,iy+diy+1,iz+diz+1]+Ψ[ix+dix+1,iy+diy+1,iz+diz+1])
    @inline Ψ_ayz(dix,diy,diz) = 0.25*(Ψ[ix+dix+1,iy+diy  ,iz+diz  ]+Ψ[ix+dix+1,iy+diy+1,iz+diz  ]+
                                       Ψ[ix+dix+1,iy+diy  ,iz+diz+1]+Ψ[ix+dix+1,iy+diy+1,iz+diz+1])
    @inline isin(A) = checkbounds(Bool,A,ix,iy,iz)
    # cell centers
    @inbounds if isin(wt.c)
        Ψs = Vec{8}(Ψ[ix,iy,iz  ],Ψ[ix+1,iy,iz  ],Ψ[ix,iy+1,iz  ],Ψ[ix+1,iy+1,iz  ],
                    Ψ[ix,iy,iz+1],Ψ[ix+1,iy,iz+1],Ψ[ix,iy+1,iz+1],Ψ[ix+1,iy+1,iz+1])
        wt.c[ix,iy,iz] = volfrac(cell,Ψs)/ω
    end
    # x faces
    @inbounds if isin(wt.x)
        Ψs = Vec{8}(Ψ_ax(0,0,0),Ψ_ax(1,0,0),Ψ_ax(0,1,0),Ψ_ax(1,1,0),
                    Ψ_ax(0,0,1),Ψ_ax(1,0,1),Ψ_ax(0,1,1),Ψ_ax(1,1,1))
        wt.x[ix,iy,iz] = volfrac(cell,Ψs)/ω
    end
    # y faces
    @inbounds if isin(wt.y)
        Ψs = Vec{8}(Ψ_ay(0,0,0),Ψ_ay(1,0,0),Ψ_ay(0,1,0),Ψ_ay(1,1,0),
                    Ψ_ay(0,0,1),Ψ_ay(1,0,1),Ψ_ay(0,1,1),Ψ_ay(1,1,1))
        wt.y[ix,iy,iz] = volfrac(cell,Ψs)/ω
    end
    # z faces
    @inbounds if isin(wt.z)
        Ψs = Vec{8}(Ψ_az(0,0,0),Ψ_az(1,0,0),Ψ_az(0,1,0),Ψ_az(1,1,0),
                    Ψ_az(0,0,1),Ψ_az(1,0,1),Ψ_az(0,1,1),Ψ_az(1,1,1))
        wt.z[ix,iy,iz] = volfrac(cell,Ψs)/ω
    end
    # xy edges
    @inbounds if isin(wt.xy)
        Ψs = Vec{8}(Ψ_axy(0,0,0),Ψ_axy(1,0,0),Ψ_axy(0,1,0),Ψ_axy(1,1,0),
                    Ψ_axy(0,0,1),Ψ_axy(1,0,1),Ψ_axy(0,1,1),Ψ_axy(1,1,1))
        wt.xy[ix,iy,iz] = volfrac(cell,Ψs)/ω
    end
    # xz edges
    @inbounds if isin(wt.xz)
        Ψs = Vec{8}(Ψ_axz(0,0,0),Ψ_axz(1,0,0),Ψ_axz(0,1,0),Ψ_axz(1,1,0),
                    Ψ_axz(0,0,1),Ψ_axz(1,0,1),Ψ_axz(0,1,1),Ψ_axz(1,1,1))
        wt.xz[ix,iy,iz] = volfrac(cell,Ψs)/ω
    end
    # yz edges
    @inbounds if isin(wt.yz)
        Ψs = Ψs = Vec{8}(Ψ_ayz(0,0,0),Ψ_ayz(1,0,0),Ψ_ayz(0,1,0),Ψ_ayz(1,1,0),
                         Ψ_ayz(0,0,1),Ψ_ayz(1,0,1),Ψ_ayz(0,1,1),Ψ_ayz(1,1,1))
        wt.yz[ix,iy,iz] = volfrac(cell,Ψs)/ω
    end
    return
end