# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@Time    :   2023/12/23 20:44:53
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import os
import shutil
import numpy as np
import pickle
import paddle
import paddle.distributed as dist
from pplsd.apis import TimerDecorator


def tensor2numpy(data):
    """
    Convert tensor to numpy

    Args:
        data (tensor|dict|list): tensor data

    Returns:
        data (numpy|dict|list): numpy data
    """
    if isinstance(data, list):
        return [tensor2numpy(val) for val in data]
    elif isinstance(data, dict):
        return {key: tensor2numpy(val) for key, val in data.items()}
    elif isinstance(data, paddle.Tensor):
        return data.numpy()
    else:
        return data


def mkdir_or_exist(dir_name, mode=0o777):
    """mkdir if not exist"""
    if dir_name == "":
        return
    dir_name = os.path.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


@TimerDecorator()
def collect_results_cpu(part_results, size, tmpdir=".dist_test"):
    """Collect results from all ranks"""
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size <= 1:
        return part_results[:size]

    # create a tmp dir
    mkdir_or_exist(tmpdir)

    # dump the part result to the dir
    paddle.save(
        part_results, os.path.join(tmpdir, "part_{}.pdparams".format(rank))
    )

    # synchronize all processes
    dist.barrier()

    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        results = []
        for i in range(world_size):
            part_file = os.path.join(tmpdir, "part_{}.pdparams".format(i))
            results.extend(paddle.load(part_file))

        # the dataloader may pad some samples
        ordered_results = results[:size]

        # sort by sample_idx
        ordered_results = sorted(
            ordered_results, key=lambda x: x["sample_idx"]
        )

        # remove tmp dir
        shutil.rmtree(tmpdir)

        return ordered_results


@TimerDecorator()
def split_results_cpu(results, tmpdir=".dist_test"):
    """split results to all ranks"""
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size <= 1:
        return results

    # create a tmp dir
    mkdir_or_exist(tmpdir)

    if rank == 0:
        # split results
        part_results_list = []
        indices = np.linspace(0, len(results), world_size + 1).astype("int32")
        for i in range(world_size):
            part_results_list.append(results[indices[i] : indices[i + 1]])

        # dump the part results to the dir
        for i in range(world_size):
            part_file = os.path.join(tmpdir, "part_{}.pdparams".format(i))
            paddle.save(part_results_list[i], part_file)

    # synchronize all processes
    dist.barrier()

    # load the part results from the dir
    part_file = os.path.join(tmpdir, "part_{}.pdparams".format(rank))
    part_results = paddle.load(part_file)

    # synchronize all processes
    dist.barrier()

    # remove tmp dir
    shutil.rmtree(tmpdir)

    return part_results


@TimerDecorator()
def collect_object(object, tmpdir=None):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size <= 1:
        return [object]

    if tmpdir is None:
        # dump result part to tensor with pickle
        part_tensor = paddle.to_tensor(
            np.frombuffer(pickle.dumps(object), dtype=np.uint8)
        )
        # gather all result part tensor shape
        shape_tensor = paddle.to_tensor(part_tensor.shape)
        shape_list = [shape_tensor.clone() for _ in range(world_size)]
        dist.all_gather(shape_list, shape_tensor)
        # padding result part tensor to max length
        shape_max = paddle.to_tensor(shape_list).max()
        # __setitem__ are not supported with dtype uint8, so we cast to int32
        part_tensor = part_tensor.cast("int32")
        part_send = paddle.zeros([shape_max], dtype="int32")
        part_send[: shape_tensor[0]] = part_tensor
        part_recv_list = [
            paddle.zeros([shape_max], dtype=part_tensor.dtype)
            for _ in range(world_size)
        ]
        # gather all result part
        dist.all_gather(part_recv_list, part_send)

        object_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            object_list.append(
                pickle.loads(recv[: shape[0]].cast("uint8").numpy().tobytes())
            )
    else:
        # create a tmp dir
        mkdir_or_exist(tmpdir)

        # dump the object to the dir
        with open(os.path.join(tmpdir, "part_{}.pkl".format(rank)), "wb") as f:
            pickle.dump(object, f)

        # synchronize all processes
        dist.barrier()

        # load results of all parts from tmp dir
        object_list = []
        for i in range(world_size):
            with open(
                os.path.join(tmpdir, "part_{}.pkl".format(i)), "rb"
            ) as f:
                object_list.append(pickle.load(f))

        # synchronize all processes: wait for pickle loading to complete
        dist.barrier()

        # remove tmp dir
        shutil.rmtree(tmpdir)

    return object_list
