using MPI
MPI.Init()

comm = MPI.COMM_WORLD
me   = MPI.Comm_rank(comm)
println("Hello world, I am $(me) of $(MPI.Comm_size(comm))")
MPI.Barrier(comm)

#println("hello")

