import numpy
from mpi4py import MPI


def setup_mpi():
    """
    Setup MPI
    :return: comm, size, rank
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
    np_list = [np // size + (1 if x < np % size else 0) for x in range(size)]
    return np_list[rank]


def gather_rho(grids, comm, ng):
    """
    Gather rho from all processors
    :param grids: grid list
    :param comm: mpi comm
    :param ng: number of grids
    :return:
    """
    global_rho = numpy.zeros(ng)
    comm.Allreduce(grids.rho, global_rho)
    grids.rho = global_rho


def gather_j(grids, comm, ng):
    """
    Gather j from all processors
    :param grids: grid list
    :param comm: mpi comm
    :param ng: number of grids
    :return:
    """
    global_jy_old = numpy.zeros(ng)
    global_jy = numpy.zeros(ng)
    global_jz_old = numpy.zeros(ng)
    global_jz = numpy.zeros(ng)
    comm.Allreduce(grids.jy_old, global_jy_old)
    comm.Allreduce(grids.jy, global_jy)
    comm.Allreduce(grids.jz_old, global_jz_old)
    comm.Allreduce(grids.jz, global_jz)
    grids.jy_old = global_jy_old - numpy.mean(global_jy_old)
    grids.jy = global_jy - numpy.mean(global_jy)
    grids.jz_old = global_jz_old - numpy.mean(global_jz_old)
    grids.jz = global_jz - numpy.mean(global_jz)



