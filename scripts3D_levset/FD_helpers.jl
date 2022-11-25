macro get_thread_idx() esc(:( begin
    ix = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x;
    iy = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y;
    iz = (workgroupIdx().z - 1) * workgroupDim().z + workitemIdx().z;
    ixi,iyi,izi = ix+1,iy+1,iz+1
    end ))
end

macro     d_xa(A)   esc(:( $A[ix+1,iy  ,iz  ] - $A[ix  ,iy  ,iz  ] )) end
macro     d_ya(A)   esc(:( $A[ix  ,iy+1,iz  ] - $A[ix  ,iy  ,iz  ] )) end
macro     d_za(A)   esc(:( $A[ix  ,iy  ,iz+1] - $A[ix  ,iy  ,iz  ] )) end
macro     d_xi(A)   esc(:( $A[ix+1,iyi ,izi ] - $A[ix  ,iyi ,izi ] )) end
macro     d_yi(A)   esc(:( $A[ixi ,iy+1,izi ] - $A[ixi ,iy  ,izi ] )) end
macro     d_zi(A)   esc(:( $A[ixi ,iyi ,iz+1] - $A[ixi ,iyi ,iz  ] )) end
macro    d2_xi(A)   esc(:( ($A[ixi+1,iyi  ,izi  ] - $A[ixi ,iyi ,izi ])  -  ($A[ixi ,iyi ,izi ] - $A[ixi-1,iyi  ,izi  ]) )) end
macro    d2_yi(A)   esc(:( ($A[ixi  ,iyi+1,izi  ] - $A[ixi ,iyi ,izi ])  -  ($A[ixi ,iyi ,izi ] - $A[ixi  ,iyi-1,izi  ]) )) end
macro    d2_zi(A)   esc(:( ($A[ixi  ,iyi  ,izi+1] - $A[ixi ,iyi ,izi ])  -  ($A[ixi ,iyi ,izi ] - $A[ixi  ,iyi  ,izi-1]) )) end
macro      all(A)   esc(:( $A[ix  ,iy  ,iz  ] )) end
macro      inn(A)   esc(:( $A[ixi ,iyi ,izi ] )) end
macro    inn_x(A)   esc(:( $A[ixi ,iy  ,iz  ] )) end
macro    inn_y(A)   esc(:( $A[ix  ,iyi ,iz  ] )) end
macro    inn_z(A)   esc(:( $A[ix  ,iy  ,izi ] )) end
macro   inn_xy(A)   esc(:( $A[ixi ,iyi ,iz  ] )) end
macro   inn_xz(A)   esc(:( $A[ixi ,iy  ,izi ] )) end
macro   inn_yz(A)   esc(:( $A[ix  ,iyi ,izi ] )) end
macro       av(A)   esc(:(($A[ix  ,iy  ,iz  ] + A[ix+1,iy  ,iz  ] +
                           $A[ix+1,iy+1,iz  ] + A[ix+1,iy+1,iz+1] +
                           $A[ix  ,iy+1,iz+1] + A[ix  ,iy  ,iz+1] +
                           $A[ix+1,iy  ,iz+1] + A[ix  ,iy+1,iz  ] )*0.125)) end
macro    av_xa(A)   esc(:(($A[ix  ,iy  ,iz  ] + A[ix+1,iy  ,iz  ] )*0.5 )) end
macro    av_ya(A)   esc(:(($A[ix  ,iy  ,iz  ] + A[ix  ,iy+1,iz  ] )*0.5 )) end
macro    av_za(A)   esc(:(($A[ix  ,iy  ,iz  ] + A[ix  ,iy  ,iz+1] )*0.5 )) end
macro    av_xi(A)   esc(:(($A[ix  ,iyi ,izi ] + A[ix+1,iyi ,izi ] )*0.5 )) end
macro    av_yi(A)   esc(:(($A[ixi ,iy  ,izi ] + A[ixi ,iy+1,izi ] )*0.5 )) end
macro    av_zi(A)   esc(:(($A[ixi ,iyi ,iz  ] + A[ixi ,iyi ,iz+1] )*0.5 )) end
macro   av_xya(A)   esc(:(($A[ix  ,iy  ,iz  ] + A[ix+1,iy  ,iz  ] +
                           $A[ix  ,iy+1,iz  ] + A[ix+1,iy+1,iz  ] )*0.25 )) end
macro   av_xza(A)   esc(:(($A[ix  ,iy  ,iz  ] + A[ix+1,iy  ,iz  ] +
                           $A[ix  ,iy  ,iz+1] + A[ix+1,iy  ,iz+1] )*0.25 )) end
macro   av_yza(A)   esc(:(($A[ix  ,iy  ,iz  ] + A[ix  ,iy+1,iz  ] +
                           $A[ix  ,iy  ,iz+1] + A[ix  ,iy+1,iz+1] )*0.25 )) end
macro   av_xyi(A)   esc(:(($A[ix  ,iy  ,izi ] + A[ix+1,iy  ,izi ] +
                           $A[ix  ,iy+1,izi ] + A[ix+1,iy+1,izi ] )*0.25 )) end
macro   av_xzi(A)   esc(:(($A[ix  ,iyi ,iz  ] + A[ix+1,iyi ,iz  ] +
                           $A[ix  ,iyi ,iz+1] + A[ix+1,iyi ,iz+1] )*0.25 )) end
macro   av_yzi(A)   esc(:(($A[ixi ,iy  ,iz  ] + A[ixi ,iy+1,iz  ] +
                           $A[ixi ,iy  ,iz+1] + A[ixi ,iy+1,iz+1] )*0.25 )) end
macro   maxloc(A)   esc(:( max.($A[ixi-1,iyi-1,izi-1],$A[ixi-1,iyi,izi-1],$A[ixi-1,iyi+1,izi-1],
                                $A[ixi  ,iyi-1,izi-1],$A[ixi  ,iyi,izi-1],$A[ixi  ,iyi+1,izi-1],
                                $A[ixi+1,iyi-1,izi-1],$A[ixi+1,iyi,izi-1],$A[ixi+1,iyi+1,izi-1],
                                $A[ixi-1,iyi-1,izi-1],$A[ixi-1,iyi,izi-1],$A[ixi-1,iyi+1,izi  ],
                                $A[ixi  ,iyi-1,izi-1],$A[ixi  ,iyi,izi-1],$A[ixi  ,iyi+1,izi  ],
                                $A[ixi+1,iyi-1,izi-1],$A[ixi+1,iyi,izi-1],$A[ixi+1,iyi+1,izi  ],
                                $A[ixi-1,iyi-1,izi-1],$A[ixi-1,iyi,izi-1],$A[ixi-1,iyi+1,izi+1],
                                $A[ixi  ,iyi-1,izi-1],$A[ixi  ,iyi,izi-1],$A[ixi  ,iyi+1,izi+1],
                                $A[ixi+1,iyi-1,izi-1],$A[ixi+1,iyi,izi-1],$A[ixi+1,iyi+1,izi+1]) )) end
