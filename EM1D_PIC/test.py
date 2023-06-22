from mpi4py import MPI
import numpy
import random

def set_mpi():
    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    rank = comm.Get_rank()
    return comm, nproc, rank
comm, nproc, rank = set_mpi()
a = numpy.array([random.random()*1, random.random()*2, random.random()*3])
if rank == 2:
    print(a)
rho = numpy.zeros(len(a))
comm.Allreduce(a, rho)
if rank == 2:
    print(rho)