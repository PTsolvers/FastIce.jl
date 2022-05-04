import ParallelStencil: INDICES
ix,iy,iz   = INDICES[1], INDICES[2], INDICES[3]
ixi,iyi,izi = :($ix+1), :($iy+1), :($iz+1)

macro define_indices(ix,iy,iz) esc(:( ($(INDICES[1]), $(INDICES[2]), $(INDICES[3])) = ($ix, $iy, $iz) )) end

macro within_x(ϕ, expr)
    bnd_chk = :( ix <= size($ϕ,1)-1 && iy <= size($ϕ,2) && iz <= size($ϕ,3) )
    esc(:( if $bnd_chk; $expr end ))
end

macro within_y(ϕ, expr)
    bnd_chk = :( ix <= size($ϕ,1) && iy <= size($ϕ,2)-1 && iz <= size($ϕ,3) )
    esc(:( if $bnd_chk; $expr end ))
end

macro within_z(ϕ, expr)
    bnd_chk = :( ix <= size($ϕ,1) && iy <= size($ϕ,2) && iz <= size($ϕ,3)-1 )
    esc(:( if $bnd_chk; $expr end ))
end

macro within_xi(ϕ, expr)
    bnd_chk = :( ix <= size($ϕ,1)-1 && iy <= size($ϕ,2)-2 && iz <= size($ϕ,3)-2 )
    esc(:( if $bnd_chk; $expr end ))
end

macro within_yi(ϕ, expr)
    bnd_chk = :( ix <= size($ϕ,1)-2 && iy <= size($ϕ,2)-1 && iz <= size($ϕ,3)-2 )
    esc(:( if $bnd_chk; $expr end ))
end

macro within_zi(ϕ, expr)
    bnd_chk = :( ix <= size($ϕ,1)-2 && iy <= size($ϕ,2)-2 && iz <= size($ϕ,3)-1 )
    esc(:( if $bnd_chk; $expr end ))
end

macro in_phase(ϕ, ph, expr)
    bnd_chk = :( ix <= size($ϕ,1) && iy <= size($ϕ,2) && iz <= size($ϕ,3) )
    esc(:( if $bnd_chk && $ϕ[ix,iy,iz] == $ph; $expr end ))
end

macro in_phases_x(ϕ, p1, p2, expr)
    bnd_chk = :( ix <= size($ϕ,1)-1 && iy <= size($ϕ,2) && iz <= size($ϕ,3) )
    esc(:( if $bnd_chk && (($ϕ[ix,iy,iz] == $p1 && $ϕ[ix+1,iy,iz] == $p2) || ($ϕ[ix,iy,iz] == $p2 && $ϕ[ix+1,iy,iz] == $p1)); $expr end ))
end

macro in_phases_xi(ϕ, p1, p2, expr)
    bnd_chk = :( ix <= size($ϕ,1)-1 && iy <= size($ϕ,2)-2 && iz <= size($ϕ,3)-2 )
    esc(:( if $bnd_chk && (($ϕ[ix,iy+1,iz+1] == $p1 && $ϕ[ix+1,iy+1,iz+1] == $p2) || ($ϕ[ix,iy+1,iz+1] == $p2 && $ϕ[ix+1,iy+1,iz+1] == $p1)); $expr end ))
end

macro in_phases_y(ϕ, p1, p2, expr)
    bnd_chk = :( ix <= size($ϕ,1) && iy <= size($ϕ,2)-1 && iz <= size($ϕ,3) )
    esc(:( if $bnd_chk && (($ϕ[ix,iy,iz] == $p1 && $ϕ[ix,iy+1,iz] == $p2) || ($ϕ[ix,iy,iz] == $p2 && $ϕ[ix,iy+1,iz] == $p1)); $expr end ))
end

macro in_phases_yi(ϕ, p1, p2, expr)
    bnd_chk = :( ix <= size($ϕ,1)-2 && iy <= size($ϕ,2)-1 && iz <= size($ϕ,3)-2 )
    esc(:( if $bnd_chk && (($ϕ[ix+1,iy,iz+1] == $p1 && $ϕ[ix+1,iy+1,iz+1] == $p2) || ($ϕ[ix+1,iy,iz+1] == $p1 && $ϕ[ix+1,iy+1,iz+1] == $p2)); $expr end ))
end

macro in_phases_z(ϕ, p1, p2, expr)
    bnd_chk = :( ix <= size($ϕ,1) && iy <= size($ϕ,2) && iz <= size($ϕ,3)-1 )
    esc(:( if $bnd_chk && (($ϕ[ix,iy,iz] == $p1 && $ϕ[ix,iy,iz+1] == $p2) || ($ϕ[ix,iy,iz] == $p2 && $ϕ[ix,iy,iz+1] == $p1)); $expr end ))
end

macro in_phases_zi(ϕ, p1, p2, expr)
    bnd_chk = :( ix <= size($ϕ,1)-2 && iy <= size($ϕ,2)-2 && iz <= size($ϕ,3)-1 )
    esc(:( if $bnd_chk && (($ϕ[ix+1,iy+1,iz] == $p1 && $ϕ[ix+1,iy+1,iz+1] == $p2) || ($ϕ[ix+1,iy+1,iz] == $p1 && $ϕ[ix+1,iy+1,iz+1] == $p2)); $expr end ))
end

macro in_phases_xy(ϕ, p1, p2, p3, p4, expr)
    bnd_chk = :( ix <= size($ϕ,1)-1 && iy <= size($ϕ,2)-1 && iz <= size($ϕ,3)-2 )
    esc(:( if $bnd_chk && $ϕ[ix,iy,iz+1] == $p1 && $ϕ[ix+1,iy,iz+1] == $p2 && $ϕ[ix,iy+1,iz+1] == $p3 && $ϕ[ix+1,iy+1,iz+1] == $p4; $expr end ))
end

macro in_phases_xz(ϕ, p1, p2, p3, p4, expr)
    bnd_chk = :( ix <= size($ϕ,1)-1 && iy <= size($ϕ,2)-2 && iz <= size($ϕ,3)-1 )
    esc(:( if $bnd_chk && $ϕ[ix,iy+1,iz] == $p1 && $ϕ[ix+1,iy+1,iz] == $p2 && $ϕ[ix,iy+1,iz+1] == $p3 && $ϕ[ix+1,iy+1,iz+1] == $p4; $expr end ))
end

macro in_phases_yz(ϕ, p1, p2, p3, p4, expr)
    bnd_chk = :( ix <= size($ϕ,1)-2 && iy <= size($ϕ,2)-1 && iz <= size($ϕ,3)-1 )
    esc(:( if $bnd_chk && $ϕ[ix+1,iy,iz] == $p1 && $ϕ[ix+1,iy+1,iz] == $p2 && $ϕ[ix+1,iy,iz+1] == $p3 && $ϕ[ix+1,iy+1,iz+1] == $p4; $expr end ))
end

macro not_in_phases_x(ϕ, p1, p2, expr)
    bnd_chk = :( ix <= size($ϕ,1)-1 && iy <= size($ϕ,2) && iz <= size($ϕ,3) )
    esc(:( if $bnd_chk && (($ϕ[ix,iy,iz] != $p1 && $ϕ[ix+1,iy,iz] != $p2) || ($ϕ[ix,iy,iz] != $p2 && $ϕ[ix+1,iy,iz] != $p1)); $expr end ))
end

macro not_in_phases_xi(ϕ, p1, p2, expr)
    bnd_chk = :( ix <= size($ϕ,1)-1 && iy <= size($ϕ,2)-2 && iz <= size($ϕ,3)-2 )
    esc(:( if $bnd_chk && (($ϕ[ix,iy+1,iz+1] != $p1 && $ϕ[ix+1,iy+1,iz+1] != $p2) || ($ϕ[ix,iy+1,iz+1] != $p2 && $ϕ[ix+1,iy+1,iz+1] != $p1)); $expr end ))
end

macro not_in_phases_y(ϕ, p1, p2, expr)
    bnd_chk = :( ix <= size($ϕ,1) && iy <= size($ϕ,2)-1 && iz <= size($ϕ,3) )
    esc(:( if $bnd_chk && (($ϕ[ix,iy,iz] != $p1 && $ϕ[ix,iy+1,iz] != $p2) || ($ϕ[ix,iy,iz] != $p2 && $ϕ[ix,iy+1,iz] != $p1)); $expr end ))
end

macro not_in_phases_yi(ϕ, p1, p2, expr)
    bnd_chk = :( ix <= size($ϕ,1)-2 && iy <= size($ϕ,2)-1 && iz <= size($ϕ,3)-2 )
    esc(:( if $bnd_chk && (($ϕ[ix+1,iy,iz+1] != $p1 && $ϕ[ix+1,iy+1,iz+1] != $p2) || ($ϕ[ix+1,iy,iz+1] != $p2 && $ϕ[ix+1,iy+1,iz+1] != $p1)); $expr end ))
end

macro not_in_phases_z(ϕ, p1, p2, expr)
    bnd_chk = :( ix <= size($ϕ,1) && iy <= size($ϕ,2) && iz <= size($ϕ,3)-1 )
    esc(:( if $bnd_chk && (($ϕ[ix,iy,iz] != $p1 && $ϕ[ix,iy,iz+1] != $p2) || ($ϕ[ix,iy,iz] != $p2 && $ϕ[ix,iy,iz+1] != $p1)); $expr end ))
end

macro not_in_phases_zi(ϕ, p1, p2, expr)
    bnd_chk = :( ix <= size($ϕ,1)-2 && iy <= size($ϕ,2)-2 && iz <= size($ϕ,3)-1 )
    esc(:( if $bnd_chk && (($ϕ[ix+1,iy+1,iz] != $p1 && $ϕ[ix+1,iy+1,iz+1] != $p2) || ($ϕ[ix+1,iy+1,iz] != $p2 && $ϕ[ix+1,iy+1,iz+1] != $p1)); $expr end ))
end
