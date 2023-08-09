macro all(A) esc(:($A[ix, iy, iz])) end

macro inn(A)   esc(:($A[ix+1, iy+1, iz+1])) end

macro inn_x(A) esc(:($A[ix+1, iy  , iz  ])) end
macro inn_y(A) esc(:($A[ix  , iy+1, iz  ])) end
macro inn_z(A) esc(:($A[ix  , iy  , iz+1])) end

macro inn_xy(A) esc(:($A[ix+1, iy+1, iz  ])) end
macro inn_xz(A) esc(:($A[ix+1, iy  , iz+1])) end
macro inn_yz(A) esc(:($A[ix  , iy+1, iz+1])) end

macro ∂_x(A) esc(:($A[ix+1, iy, iz] - $A[ix, iy, iz])) end
macro ∂_y(A) esc(:($A[ix, iy+1, iz] - $A[ix, iy, iz])) end
macro ∂_z(A) esc(:($A[ix, iy, iz+1] - $A[ix, iy, iz])) end

macro ∂_xi(A) esc(:($A[ix+1, iy+1, iz+1] - $A[ix, iy+1, iz+1])) end
macro ∂_yi(A) esc(:($A[ix+1, iy+1, iz+1] - $A[ix+1, iy, iz+1])) end
macro ∂_zi(A) esc(:($A[ix+1, iy+1, iz+1] - $A[ix+1, iy+1, iz])) end

macro av_xyi(A) esc(:(0.25 * ($A[ix, iy, iz+1] + $A[ix+1, iy, iz+1] + $A[ix+1, iy+1, iz+1] + $A[ix, iy+1, iz+1]))) end
macro av_xzi(A) esc(:(0.25 * ($A[ix, iy+1, iz] + $A[ix+1, iy+1, iz] + $A[ix+1, iy+1, iz+1] + $A[ix, iy+1, iz+1]))) end
macro av_yzi(A) esc(:(0.25 * ($A[ix+1, iy, iz] + $A[ix+1, iy+1, iz] + $A[ix+1, iy+1, iz+1] + $A[ix+1, iy, iz+1]))) end

macro av_xy(A) esc(:(0.25 * ($A[ix, iy, iz] + $A[ix+1, iy, iz] + $A[ix+1, iy+1, iz] + $A[ix, iy+1, iz]))) end
macro av_xz(A) esc(:(0.25 * ($A[ix, iy, iz] + $A[ix+1, iy, iz] + $A[ix+1, iy, iz+1] + $A[ix, iy, iz+1]))) end
macro av_yz(A) esc(:(0.25 * ($A[ix, iy, iz] + $A[ix, iy+1, iz] + $A[ix, iy+1, iz+1] + $A[ix, iy, iz+1]))) end