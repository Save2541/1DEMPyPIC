import numpy
from mpi4py import MPI

from . import user_input


def setup_mpi():
    """
    Setup MPI
    :param ng: number of grids
    :return: comm, size, rank, global_array
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    return comm, size, rank


def get_ranked_np(np, size, rank):
    """
    Get number of particles for a processor
    :param np: number of particles
    :param size: number of processors
    :param rank: processor rank
    :return: number of particles in the processor
    """
    np_per_rank = np // size
    np_list = [np_per_rank + (1 if x < np % size else 0) for x in range(size)]
    return np_list[rank]


basket = numpy.zeros(user_input.ng)


def gather_rho(grids, comm):
    """
    Gather rho from all processors
    :param grids: grid list
    :param comm: mpi comm
    :return:
    """
    comm.Allreduce(grids.rho, basket)
    grids.rho = basket


basket_jy_old = numpy.zeros(user_input.ng)
basket_jz_old = numpy.zeros(user_input.ng)
basket_jy = numpy.zeros(user_input.ng)
basket_jz = numpy.zeros(user_input.ng)


def gather_j(grids, comm):
    """
    Gather j from all processors
    :param grids: grid list
    :param comm: mpi comm
    :return:
    """
    # basket = numpy.zeros_like(basket)
    comm.Allreduce(grids.jy_old, basket_jy_old)
    grids.jy_old = basket_jy_old - numpy.mean(basket_jy_old)
    # basket = numpy.zeros_like(basket)
    comm.Allreduce(grids.jy, basket_jy)
    grids.jy = basket_jy - numpy.mean(basket_jy)
    # basket = numpy.zeros_like(basket)
    comm.Allreduce(grids.jz_old, basket_jz_old)
    grids.jz_old = basket_jz_old - numpy.mean(basket_jz_old)
    # basket = numpy.zeros_like(basket)
    comm.Allreduce(grids.jz, basket_jz)
    grids.jz = basket_jz - numpy.mean(basket_jz)
