"""
Helpers for distributed training.
"""

import io
import os
import socket
import blobfile as bf
import torch as th
import torch.distributed as dist

# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8
SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group using SLURM environment variables.
    """
    if dist.is_initialized():
        return

    # SLURM-specific environment variables
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    master_addr = os.environ["SLURM_NODELIST"].split(",")[0]
    master_port = _find_free_port()

    # Configure PyTorch distributed
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    backend = "gloo" if not th.cuda.is_available() else "nccl"
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across ranks.
    """
    rank = dist.get_rank()
    if rank == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // (2**30)
        if len(data) % (2**30):
            num_chunks += 1
        dist.broadcast_object_list([num_chunks])
        for i in range(0, len(data), 2**30):
            dist.broadcast_object_list([data[i : i + 2**30]])
    else:
        num_chunks = dist.broadcast_object_list([None])[0]
        data = bytearray()
        for _ in range(num_chunks):
            data_chunk = dist.broadcast_object_list([None])[0]
            data.extend(data_chunk)

    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    """
    Find an available port for communication.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
