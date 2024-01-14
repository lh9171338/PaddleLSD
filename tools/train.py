# -*- encoding: utf-8 -*-
"""
@File    :   train.py
@Time    :   2023/12/16 23:38:54
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import os
import paddle
from functools import partial
import random
import numpy as np
import logging
import argparse
import time
from lh_pplsd.apis import Config, Trainer


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument(
        "--config",
        help="The config file",
        type=str,
    )
    parser.add_argument(
        "--eval_interval",
        help="Eval at every eval_interval while training",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--log_interval",
        help="Display logging information at every log_interval",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--num_workers",
        help="Num workers for data loader",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--resume",
        help="Whether to resume training from checkpoint",
        action="store_true",
    )
    parser.add_argument(
        "--model",
        help="Pretrained parameters of the model",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--save_dir",
        help="The directory for saving the model snapshot",
        default="./output/temp",
        type=str,
    )
    parser.add_argument(
        "--save_interval",
        help="How many epochs to save a model snapshot once during training",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--keep_checkpoint_max",
        help="Maximum number of checkpoints to save",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--seed",
        help="Set the random seed of paddle during training",
        default=1234,
        type=int,
    )
    args = parser.parse_args()
    print(args)

    return args


def set_seed(seed):
    """set seed"""
    logging.info("use random seed {}".format(seed))
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def worker_init_fn(worker_id, num_workers, rank, seed):
    """worker init fn"""
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main(args):
    """main"""
    if args.seed is not None:
        set_seed(args.seed)

    if args.config is None:
        raise RuntimeError("No configuration file specified!")

    if not os.path.exists(args.config):
        raise RuntimeError(
            "Config file `{}` does not exist!".format(args.config)
        )

    cfg = Config(path=args.config)
    logging.info(cfg)

    if args.model is not None:
        state_dict = paddle.load(args.model)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        cfg.model.set_state_dict(state_dict)

    if cfg.train_dataset is None:
        raise RuntimeError(
            "The training dataset is not specified in the configuration file!"
        )
    elif len(cfg.train_dataset) == 0:
        raise ValueError(
            "The length of training dataset is 0. Please check if your dataset is valid!"
        )

    dic = cfg.to_dict()

    rank = paddle.distributed.get_rank()
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
    init_fn = (
        partial(
            worker_init_fn,
            num_workers=args.num_workers,
            rank=rank,
            seed=args.seed,
        )
        if args.seed is not None
        else None
    )

    dic.update(
        {
            "resume": args.resume,
            "save_dir": args.save_dir,
            "save_interval": args.save_interval,
            "keep_checkpoint_max": args.keep_checkpoint_max,
            "eval_interval": args.eval_interval,
            "log_interval": args.log_interval,
            "dataloader_fn": {
                "batch_size": dic["batch_size"],
                "num_workers": args.num_workers,
                "worker_init_fn": init_fn,
            },
        }
    )

    trainer = Trainer(**dic)
    trainer.train()


if __name__ == "__main__":
    # set base logging config
    fmt = "[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)

    t1 = time.time()

    args = parse_args()
    main(args)

    t2 = time.time()
    logging.info("time: {}".format(t2 - t1))
