@tiny function _kernel_init_level_set!(Ψ,dem,dem_grid,Ψ_grid,cutoff,R)
    ix,iy,iz = @indices
    x,y,z    = Ψ_grid[1][ix],Ψ_grid[2][iy],Ψ_grid[3][iz]
    P        = R*Point3(x,y,z)
    ud,sgn   = sd_dem(P,cutoff,dem,dem_grid)
    @inbounds Ψ[ix,iy,iz] = ud*sgn
    return
end

@tiny function _kernel_compute_dΨ_dt!(dΨ_dt,Ψ,Ψ0,dx,dy,dz)
    ix,iy,iz = @indices
    @inline changes_sign_x(disp) = @inbounds Ψ0[ix,iy,iz]*Ψ0[ix+disp,iy,iz] < 0
    @inline changes_sign_y(disp) = @inbounds Ψ0[ix,iy,iz]*Ψ0[ix,iy+disp,iz] < 0
    @inline changes_sign_z(disp) = @inbounds Ψ0[ix,iy,iz]*Ψ0[ix,iy,iz+disp] < 0
    ch_x, ch_y, ch_z = false, false, false
    ∂Ψ0_∂x, ∂Ψ0_∂y, ∂Ψ0_∂z = 0.0, 0.0, 0.0
    if ix ∈ axes(Ψ0,1)[2:end-1]
        ch_x = changes_sign_x(1) || changes_sign_x(-1)
        @inbounds ∂Ψ0_∂x = (Ψ0[ix+1,iy,iz]-Ψ0[ix-1,iy,iz])/(2dx)
    end
    if iy ∈ axes(Ψ0,2)[2:end-1]
        ch_y = changes_sign_y(1) || changes_sign_y(-1)
        @inbounds ∂Ψ0_∂y = (Ψ0[ix,iy+1,iz]-Ψ0[ix,iy-1,iz])/(2dy)
    end
    if iz ∈ axes(Ψ0,3)[2:end-1]
        ch_z = changes_sign_z(1) || changes_sign_z(-1)
        @inbounds ∂Ψ0_∂z = (Ψ0[ix,iy,iz+1]-Ψ0[ix,iy,iz-1])/(2dz)
    end
    if (ch_x || ch_y || ch_z)
        # local surface reconstruction
        @inbounds D = Ψ0[ix,iy,iz]/sqrt(∂Ψ0_∂x^2 + ∂Ψ0_∂y^2 + ∂Ψ0_∂z^2)
        @inbounds dΨ_dt[ix,iy,iz] = (D-sign(Ψ0[ix,iy,iz])*abs(Ψ[ix,iy,iz]))/dx
    else
        @inbounds begin
            # Hamilton-Jacobi with Godunov flux
            # direction '-' derivatives
            ∂Ψ_∂x⁻ = ix > 1 ? (Ψ[ix,iy,iz] - Ψ[ix-1,iy,iz])/dx : 0.0
            ∂Ψ_∂y⁻ = iy > 1 ? (Ψ[ix,iy,iz] - Ψ[ix,iy-1,iz])/dy : 0.0
            ∂Ψ_∂z⁻ = iz > 1 ? (Ψ[ix,iy,iz] - Ψ[ix,iy,iz-1])/dy : 0.0
            # direction '+' derivatives
            ∂Ψ_∂x⁺ = ix < size(Ψ,1) ? (Ψ[ix+1,iy,iz] - Ψ[ix,iy,iz]) / dx : 0.0
            ∂Ψ_∂y⁺ = iy < size(Ψ,2) ? (Ψ[ix,iy+1,iz] - Ψ[ix,iy,iz]) / dy : 0.0
            ∂Ψ_∂z⁺ = iz < size(Ψ,3) ? (Ψ[ix,iy,iz+1] - Ψ[ix,iy,iz]) / dz : 0.0
            # upwind fluxes
            ∂Ψ_∂x2 = Ψ0[ix,iy,iz] >= 0 ? max(max(∂Ψ_∂x⁻,0)^2, min(∂Ψ_∂x⁺,0)^2) :
                                         max(min(∂Ψ_∂x⁻,0)^2, max(∂Ψ_∂x⁺,0)^2)
            ∂Ψ_∂y2 = Ψ0[ix,iy,iz] >= 0 ? max(max(∂Ψ_∂y⁻,0)^2, min(∂Ψ_∂y⁺,0)^2) :
                                         max(min(∂Ψ_∂y⁻,0)^2, max(∂Ψ_∂y⁺,0)^2)
            ∂Ψ_∂z2 = Ψ0[ix,iy,iz] >= 0 ? max(max(∂Ψ_∂z⁻,0)^2, min(∂Ψ_∂z⁺,0)^2) :
                                         max(min(∂Ψ_∂z⁻,0)^2, max(∂Ψ_∂z⁺,0)^2)
            # compute update
            dΨ_dt[ix,iy,iz] = sign(Ψ0[ix,iy,iz])*(1.0-sqrt(∂Ψ_∂x2+∂Ψ_∂y2+∂Ψ_∂z2))
        end
    end
    return
end

@tiny function _kernel_update_Ψ!(Ψ, dΨ_dt, dt)
    I = @cartesianindex
    @inbounds Ψ[I] += dt*dΨ_dt[I]
    return
end