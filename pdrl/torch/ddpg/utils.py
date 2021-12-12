from mpi4py import MPI


def proc_id():
    return MPI.COMM_WORLD.Get_rank()


def num_procs():
    return MPI.COMM_WORLD.Get_size()


def allreduce(*args, **kwargs):
    return MPI.COMM_WORLD.Allreduce(*args, **kwargs)
