from pdrl.utils.mpi import mpi_avg, num_procs, broadcast


def mpi_avg_grad(module):
    """MPIプロセス間の勾配の平均化"""
    if num_procs() == 1:
        return

    for p in module.parameters():
        p_grad_numpy = p.grad.numpy()
        avg_p_grad = mpi_avg(p.grad)
        p_grad_numpy[:] = avg_p_grad[:]


def sync_params(module):
    """パラメータをMPIプロセス間で同期"""
    if num_procs() == 1:
        return
    
    for p in module.parameters():
        p_numpy = p.data.numpy()
        broadcast(p_numpy)
