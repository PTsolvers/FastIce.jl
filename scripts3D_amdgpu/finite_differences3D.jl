# Finite differences 3D
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
macro       av(A)   esc(:(($A[ix  ,iy  ,iz  ] + $A[ix+1,iy  ,iz  ] +
                           $A[ix+1,iy+1,iz  ] + $A[ix+1,iy+1,iz+1] +
                           $A[ix  ,iy+1,iz+1] + $A[ix  ,iy  ,iz+1] +
                           $A[ix+1,iy  ,iz+1] + $A[ix  ,iy+1,iz  ] )*0.125)) end
macro    av_xa(A)   esc(:(($A[ix  ,iy  ,iz  ] + $A[ix+1,iy  ,iz  ] )*0.5 )) end
macro    av_ya(A)   esc(:(($A[ix  ,iy  ,iz  ] + $A[ix  ,iy+1,iz  ] )*0.5 )) end
macro    av_za(A)   esc(:(($A[ix  ,iy  ,iz  ] + $A[ix  ,iy  ,iz+1] )*0.5 )) end
macro    av_xi(A)   esc(:(($A[ix  ,iyi ,izi ] + $A[ix+1,iyi ,izi ] )*0.5 )) end
macro    av_yi(A)   esc(:(($A[ixi ,iy  ,izi ] + $A[ixi ,iy+1,izi ] )*0.5 )) end
macro    av_zi(A)   esc(:(($A[ixi ,iyi ,iz  ] + $A[ixi ,iyi ,iz+1] )*0.5 )) end
macro   av_xya(A)   esc(:(($A[ix  ,iy  ,iz  ] + $A[ix+1,iy  ,iz  ] +
                           $A[ix  ,iy+1,iz  ] + $A[ix+1,iy+1,iz  ] )*0.25 )) end
macro   av_xza(A)   esc(:(($A[ix  ,iy  ,iz  ] + $A[ix+1,iy  ,iz  ] +
                           $A[ix  ,iy  ,iz+1] + $A[ix+1,iy  ,iz+1] )*0.25 )) end
macro   av_yza(A)   esc(:(($A[ix  ,iy  ,iz  ] + $A[ix  ,iy+1,iz  ] +
                           $A[ix  ,iy  ,iz+1] + $A[ix  ,iy+1,iz+1] )*0.25 )) end
macro   av_xyi(A)   esc(:(($A[ix  ,iy  ,izi ] + $A[ix+1,iy  ,izi ] +
                           $A[ix  ,iy+1,izi ] + $A[ix+1,iy+1,izi ] )*0.25 )) end
macro   av_xzi(A)   esc(:(($A[ix  ,iyi ,iz  ] + $A[ix+1,iyi ,iz  ] +
                           $A[ix  ,iyi ,iz+1] + $A[ix+1,iyi ,iz+1] )*0.25 )) end
macro   av_yzi(A)   esc(:(($A[ixi ,iy  ,iz  ] + $A[ixi ,iy+1,iz  ] +
                           $A[ixi ,iy  ,iz+1] + $A[ixi ,iy+1,iz+1] )*0.25 )) end
macro     harm(A)  esc(:(1.0/(1.0/$A[ix  ,iy  ,iz  ] + 1.0/$A[ix+1,iy  ,iz  ] +
                              1.0/$A[ix+1,iy+1,iz  ] + 1.0/$A[ix+1,iy+1,iz+1] +
                              1.0/$A[ix  ,iy+1,iz+1] + 1.0/$A[ix  ,iy  ,iz+1] +
                              1.0/$A[ix+1,iy  ,iz+1] + 1.0/$A[ix  ,iy+1,iz  ] )*8.0 )) end
macro  harm_xa(A)  esc(:(1.0/(1.0/$A[ix  ,iy  ,iz  ] + 1.0/$A[ix+1,iy  ,iz  ] )*2.0 )) end
macro  harm_ya(A)  esc(:(1.0/(1.0/$A[ix  ,iy  ,iz  ] + 1.0/$A[ix  ,iy+1,iz  ] )*2.0 )) end
macro  harm_za(A)  esc(:(1.0/(1.0/$A[ix  ,iy  ,iz  ] + 1.0/$A[ix  ,iy  ,iz+1] )*2.0 )) end
macro  harm_xi(A)  esc(:(1.0/(1.0/$A[ix  ,iyi ,izi ] + 1.0/$A[ix+1,iyi ,izi ] )*2.0 )) end
macro  harm_yi(A)  esc(:(1.0/(1.0/$A[ixi ,iy  ,izi ] + 1.0/$A[ixi ,iy+1,izi ] )*2.0 )) end
macro  harm_zi(A)  esc(:(1.0/(1.0/$A[ixi ,iyi ,iz  ] + 1.0/$A[ixi ,iyi ,iz+1] )*2.0 )) end
macro harm_xya(A)  esc(:(1.0/(1.0/$A[ix  ,iy  ,iz  ] + 1.0/$A[ix+1,iy  ,iz  ] +
                              1.0/$A[ix  ,iy+1,iz  ] + 1.0/$A[ix+1,iy+1,iz  ] )*4.0 )) end
macro harm_xza(A)  esc(:(1.0/(1.0/$A[ix  ,iy  ,iz  ] + 1.0/$A[ix+1,iy  ,iz  ] +
                              1.0/$A[ix  ,iy  ,iz+1] + 1.0/$A[ix+1,iy  ,iz+1] )*4.0 )) end
macro harm_yza(A)  esc(:(1.0/(1.0/$A[ix  ,iy  ,iz  ] + 1.0/$A[ix  ,iy+1,iz  ] +
                              1.0/$A[ix  ,iy  ,iz+1] + 1.0/$A[ix  ,iy+1,iz+1] )*4.0 )) end
macro harm_xyi(A)  esc(:(1.0/(1.0/$A[ix  ,iy  ,izi ] + 1.0/$A[ix+1,iy  ,izi ] +
                              1.0/$A[ix  ,iy+1,izi ] + 1.0/$A[ix+1,iy+1,izi ] )*4.0 )) end
macro harm_xzi(A)  esc(:(1.0/(1.0/$A[ix  ,iyi ,iz  ] + 1.0/$A[ix+1,iyi ,iz  ] +
                              1.0/$A[ix  ,iyi ,iz+1] + 1.0/$A[ix+1,iyi ,iz+1] )*4.0 )) end
macro harm_yzi(A)  esc(:(1.0/(1.0/$A[ixi ,iy  ,iz  ] + 1.0/$A[ixi ,iy+1,iz  ] +
                              1.0/$A[ixi ,iy  ,iz+1] + 1.0/$A[ixi ,iy+1,iz+1] )*4.0 )) end
macro   maxloc(A)  esc(:( max( max( max( max($A[ixi-1,iyi  ,izi  ], $A[ixi+1,iyi  ,izi  ])  , $A[ixi  ,iyi  ,izi  ] ),
                                         max($A[ixi  ,iyi-1,izi  ], $A[ixi  ,iyi+1,izi  ]) ),
                                         max($A[ixi  ,iyi  ,izi-1], $A[ixi  ,iyi  ,izi+1]) ) )) end
macro   minloc(A)  esc(:( min( min( min( min($A[ixi-1,iyi  ,izi  ], $A[ixi+1,iyi  ,izi  ])  , $A[ixi  ,iyi  ,izi  ] ),
                                         min($A[ixi  ,iyi-1,izi  ], $A[ixi  ,iyi+1,izi  ]) ),
                                         min($A[ixi  ,iyi  ,izi-1], $A[ixi  ,iyi  ,izi+1]) ) )) end
