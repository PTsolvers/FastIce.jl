using ImplicitGlobalGrid

nx, ny, nz = 6, 6, 6

me, dims, nprocs, coords, comm_cart = init_global_grid(nx, ny, nz)

A = fill(me, nx, ny, nz)

update_halo!(A)

sleep(me)
@info "me == $me"
display(A)

finalize_global_grid()