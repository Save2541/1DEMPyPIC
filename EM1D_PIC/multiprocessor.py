import numpy
from mpi4py import MPI

from . import user_input
from . import qol


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


def get_minimum(value, comm):
    """
    Get the minimum value across processors
    """
    return comm.allreduce(value, MPI.MIN)


def gather(x, comm):
    """
    Gather x to root=0 processor
    """
    return comm.gather(x, root=0)


def get_maximum(value, comm):
    """
    Get the maximum value across processors
    """
    return comm.allreduce(value, MPI.MAX)


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


basket_den = numpy.zeros((qol.get_n_sp(), user_input.ng))


def gather_den(grids, comm):
    """
    Gather number density from all processors
    :param grids: grid list
    :param comm: mpi comm
    :return:
    """
    comm.Allreduce(grids.den, basket_den)
    grids.den = basket_den


basket_jy_left = numpy.zeros(user_input.ng)
basket_jz_left = numpy.zeros(user_input.ng)
basket_jy_right = numpy.zeros(user_input.ng)
basket_jz_right = numpy.zeros(user_input.ng)


def gather_j(grids, comm):
    """
    Gather j from all processors
    :param grids: grid list
    :param comm: mpi comm
    :return:
    """
    comm.Allreduce(grids.jy_left, basket_jy_left)
    grids.jy_left = basket_jy_left
    # grids.jy_old = basket_jy_old - numpy.mean(basket_jy_old)
    comm.Allreduce(grids.jy_right, basket_jy_right)
    grids.jy_right = basket_jy_right
    # grids.jy = basket_jy - numpy.mean(basket_jy)
    comm.Allreduce(grids.jz_left, basket_jz_left)
    grids.jz_left = basket_jz_left
    # grids.jz_old = basket_jz_old - numpy.mean(basket_jz_old)
    comm.Allreduce(grids.jz_right, basket_jz_right)
    grids.jz_right = basket_jz_right
    # grids.jz = basket_jz - numpy.mean(basket_jz)


def gather_xv(output, comm, size, rank):
    """
    Gather x and v to rank=0 processor
    """
    basket_x = None
    basket_v = None
    x_data = None
    v_data = None
    if hasattr(output, "x") and hasattr(output, "v"):
        output_shape = output.x.shape
        data_shape = output_shape * numpy.array([1, 1, size])
        if rank == 0:
            basket_x = numpy.empty((size, *output_shape))
            basket_v = numpy.empty((size, *output_shape))
            x_data = numpy.zeros(data_shape)
            v_data = numpy.zeros(data_shape)
        comm.Gather(output.x, basket_x, root=0)
        comm.Gather(output.v, basket_v, root=0)
        if rank == 0:
            x_data = numpy.concatenate(basket_x, axis=-1)
            v_data = numpy.concatenate(basket_v, axis=-1)

    elif hasattr(output, "x"):
        output_shape = output.x.shape
        data_shape = output_shape * numpy.array([1, 1, size])
        if rank == 0:
            basket_x = numpy.empty((size, *output_shape))
            x_data = numpy.zeros(data_shape)
        comm.Gather(output.x, basket_x, root=0)
        if rank == 0:
            x_data = numpy.concatenate(basket_x, axis=-1)

    elif hasattr(output, "v"):
        output_shape = output.v.shape
        data_shape = output_shape * numpy.array([1, 1, size])
        if rank == 0:
            basket_v = numpy.empty((size, *output_shape))
            v_data = numpy.zeros(data_shape)
        comm.Gather(output.v, basket_v, root=0)
        if rank == 0:
            v_data = numpy.concatenate(basket_v, axis=-1)

    return x_data, v_data
