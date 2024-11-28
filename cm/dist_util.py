"""
Helpers for distributed training.
"""

import io
import os
import socket
import blobfile as bf
import torch as th
import torch.distributed as dist
import socket
from functools import partial
import os
import torch
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

# rank % GPUS_PER_NODE
GPUS_PER_NODE = 8
SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group
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
    init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def wrap_model(model):
    """
    Args:
        model: a torch.nn.Module
        rank: the rank of the DDP training process for that model
    Returns:
        a copy of that model on the device dedicated to the process rank
    """
    rank= int(os.environ["SLURM_PROCID"])
    model = DDP(model.to(rank), device_ids=[rank], output_device=rank)
    return model


def get_dataloader(dset, seed=42, micro_batch_size=1,  **kwargs):
    """
    Args:
        dset: a torch.utils.data.Dataset
        world_size: Total number of processes
        rank: Unique identifier of each process
        seed: random seed for reproducibility
              (choose a large number with balanced bits, such as 42, it's the answer to the universe, and everything)
        micro_batch_size: the batch size of the dataloader (not necessarily the global batch size for optimization)
        **kwargs: placeholder for duck-typing
    Returns:
        A distributed dataloader, that distributes samples to the respective process rank
    """
    rank = int(os.getenv("SLURM_PROCID", 0))
    world_size = int(os.getenv("SLURM_NTASKS", 1))
    sampler = DistributedSampler(
        dset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=False)
    return DataLoader(dset, batch_size=micro_batch_size, sampler=sampler)


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
            dist.broadcast_object_list([data[i: i + 2**30]])
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
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
