import numpy
from mpi4py import MPI

from . import user_input


def setup_mpi(ng=user_input.ng):
    """
    Setup MPI
    :param ng: number of grids
    :return: comm, size, rank, global_array
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    basket = numpy.zeros(ng)
    return comm, size, rank, basket


def get_ranked_np(np, size, rank):
    """
    Get number of particles for a processor
    :param np: number of particles
    :param size: number of processors
    :param rank: processor rank
    :return: number of particles in the processor
    """
    np_per_rank = np//size
    np_list = [np_per_rank + (1 if x < np % size else 0) for x in range(size)]
    return np_list[rank]


def gather_rho(grids, comm, basket):
    """
    Gather rho from all processors
    :param grids: grid list
    :param comm: mpi comm
    :param basket: reusable array for gathering
    :return:
    """
    comm.Allreduce(grids.rho, basket)
    grids.rho = basket


def gather_j(grids, comm, basket):
    """
    Gather j from all processors
    :param grids: grid list
    :param comm: mpi comm
    :param basket: reusable array for gathering
    :return:
    """
    comm.Allreduce(grids.jy_old, basket)
    grids.jy_old = basket - numpy.mean(basket)
    comm.Allreduce(grids.jy, basket)
    grids.jy = basket - numpy.mean(basket)
    comm.Allreduce(grids.jz_old, basket)
    grids.jz_old = basket - numpy.mean(basket)
    comm.Allreduce(grids.jz, basket)
    grids.jz = basket - numpy.mean(basket)



