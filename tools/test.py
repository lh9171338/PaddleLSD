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
import logging
import argparse
import time
from pplsd.apis import Config, Trainer


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description="Model testing")
    parser.add_argument(
        "--config",
        help="The config file",
        type=str,
    )
    parser.add_argument("--no_infer", help="No infer", action="store_true")
    parser.add_argument(
        "--save_result", help="Save result while testing", action="store_true"
    )
    parser.add_argument(
        "--do_eval", help="Eval while testing", action="store_true"
    )
    parser.add_argument(
        "--do_visualize", help="Visualize while testing", action="store_true"
    )
    parser.add_argument(
        "--batch_size",
        help="Batch size for testing",
        type=int,
    )
    parser.add_argument(
        "--num_workers",
        help="Num workers for data loader",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--model",
        help="Model for testing",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--save_dir",
        help="The directory for saving the model snapshot",
        default="./output/temp",
        type=str,
    )
    args = parser.parse_args()
    print(args)

    return args


def main(args):
    """main"""
    if args.config is None:
        raise RuntimeError("No configuration file specified!")

    if not os.path.exists(args.config):
        raise RuntimeError(
            "Config file `{}` does not exist!".format(args.config)
        )

    cfg = Config(path=args.config, batch_size=args.batch_size)
    logging.info(cfg)

    if args.model is None:
        raise RuntimeError("No model file specified!")

    if not os.path.exists(args.model):
        raise RuntimeError(
            "Model file `{}` does not exist!".format(args.model)
        )
    state_dict = paddle.load(args.model)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    cfg.model.set_state_dict(state_dict)

    dic = cfg.to_dict()

    rank = paddle.distributed.get_rank()
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)

    dic.update(
        {
            "save_dir": args.save_dir,
            "dataloader_fn": {
                "batch_size": dic["batch_size"],
                "num_workers": args.num_workers,
            },
        }
    )

    trainer = Trainer(**dic)
    trainer.test(
        args.save_result, args.do_eval, args.do_visualize, args.no_infer
    )


if __name__ == "__main__":
    # set base logging config
    fmt = "[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)

    t1 = time.time()

    args = parse_args()
    main(args)

    t2 = time.time()
    logging.info("time: {}".format(t2 - t1))
