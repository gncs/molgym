# The content of this file is based on: OpenAI Spinning Up https://spinningup.openai.com/.
import logging
import os
from typing import Callable, Tuple

import numpy as np
import torch
from mpi4py import MPI


def get_proc_rank() -> int:
    """Get rank of calling process."""
    return MPI.COMM_WORLD.Get_rank()


def is_main_proc() -> bool:
    """Check if the current process is either the only one or has rank 0"""
    return get_num_procs() == 1 or get_proc_rank() == 0


def get_num_procs() -> int:
    """Count active MPI processes."""
    return MPI.COMM_WORLD.Get_size()


def mpi_broadcast(x: object, root=0) -> None:
    MPI.COMM_WORLD.Bcast(x, root=root)


def set_barrier() -> None:
    MPI.COMM_WORLD.Barrier()


def mpi_reduce(x: np.ndarray, op: Callable) -> np.ndarray:
    """Apply reduce operation over MPI processes."""
    buffer = np.zeros_like(x)
    MPI.COMM_WORLD.Allreduce(x, buffer, op=op)
    return buffer


def mpi_sum(x: np.ndarray) -> np.ndarray:
    """Sum an array over MPI processes."""
    return mpi_reduce(x, op=MPI.SUM)


def mpi_avg(x: np.ndarray) -> np.ndarray:
    """Average an array over MPI processes."""
    return mpi_sum(x) / get_num_procs()


def mpi_mean_std(x: np.ndarray, axis: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Get mean and std along last dimension of array x across MPI processes."""
    local_sum = np.sum(x, axis=axis, keepdims=True)
    local_len = x.shape[axis]

    global_sum = mpi_sum(local_sum)
    global_len = mpi_sum(np.asarray(local_len))

    mean = global_sum / global_len

    local_sum_sq = np.sum(np.square(x - mean), axis=axis, keepdims=True)
    global_sum_sq = mpi_sum(local_sum_sq)
    std = np.sqrt(global_sum_sq / global_len)

    return mean, std


def sync_params(module: torch.nn.modules.Module):
    """ Sync all parameters of module across all MPI processes. """
    if get_num_procs() == 1:
        return
    for p in module.parameters():
        p_numpy = p.data.numpy()  # numpy view of tensor data (shares the same underlying storage)
        mpi_broadcast(p_numpy)


def mpi_avg_grads(module: torch.nn.modules.Module):
    """ Average contents of gradient buffers across MPI processes. """
    if get_num_procs() == 1:
        return
    for p in module.parameters():
        if p.grad is None:
            continue

        p_grad_numpy = p.grad.numpy()  # numpy view of tensor data
        avg_p_grad = mpi_avg(p_grad_numpy)
        p_grad_numpy[:] = avg_p_grad[:]


class MPIFileHandler(logging.StreamHandler):
    def __init__(self,
                 filename: str,
                 mode=MPI.MODE_WRONLY | MPI.MODE_CREATE | MPI.MODE_APPEND,
                 encoding='utf-8',
                 comm=MPI.COMM_WORLD):
        self.baseFilename = os.path.abspath(filename)
        self.mode = mode
        self.encoding = encoding
        self.comm = comm
        super().__init__(self._open())

    def _open(self) -> MPI.File:
        stream = MPI.File.Open(self.comm, self.baseFilename, self.mode)
        stream.Set_atomicity(True)
        return stream

    def emit(self, record: logging.LogRecord):
        """
        Emit a record.
        If a formatter is specified, it is used to format the record.
        The record is then written to the stream with a trailing newline.  If
        exception information is present, it is formatted using
        traceback.print_exception and appended to the stream.  If the stream
        has an 'encoding' attribute, it is used to determine how to do the
        output to the stream.

        Modification:
            stream is MPI.File, so it must use `Write_shared` method rather
            than `write` method. And `Write_shared` method only accept
            bytestring, so `encode` is used. `Write_shared` should be invoked
            only once in each all of this emit function to keep atomicity.
        """
        try:
            msg = self.format(record)
            self.stream.Write_shared((msg + self.terminator).encode(self.encoding))
        except Exception:
            self.handleError(record)

    def close(self):
        if self.stream:
            self.stream.Sync()
            self.stream.Close()
            self.stream = None
