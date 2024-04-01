# -*- encoding: utf-8 -*-
"""
@File    :   trainer.py
@Time    :   2023/12/17 13:22:37
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import os
import time
import paddle
from paddle.metric import Metric
import tqdm
import paddle.distributed as dist
import logging
from visualdl import LogWriter
from typing import Optional
from pplsd.apis import Timer
from pplsd.visualizers import Visualizer
import pplsd.apis.utils as api_utils


def default_dataloader_build_fn(**kwargs):
    """default dataloader build function"""

    def _generate_loader(dataset: paddle.io.Dataset) -> paddle.io.DataLoader:
        args = kwargs.copy()
        batch_size = args.pop("batch_size", 1)
        shuffle = dataset.is_train_mode
        drop_last = args.pop(
            "drop_last", False if not dataset.is_train_mode else True
        )
        collate_fn = getattr(dataset, "collate_fn", None)

        if dist.get_world_size() > 1:
            BatchSampler = paddle.io.DistributedBatchSampler
        else:
            BatchSampler = paddle.io.BatchSampler

        batch_sampler = BatchSampler(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        return paddle.io.DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            **args,
        )

    return _generate_loader


class Trainer:
    """
    Trainer
    """

    def __init__(
        self,
        model: paddle.nn.Layer,
        optimizer: paddle.optimizer.Optimizer,
        lr_scheduler: paddle.optimizer.lr.LRScheduler,
        scheduler_by_epoch: bool = True,
        batch_size: Optional[int] = 1,
        epochs: Optional[int] = None,
        train_dataset: Optional[paddle.io.Dataset] = None,
        val_dataset: Optional[paddle.io.Dataset] = None,
        visualizer: Optional[Visualizer] = None,
        metric: Optional[Metric] = None,
        resume: bool = False,
        save_dir: Optional[str] = None,
        save_interval: Optional[int] = None,
        keep_checkpoint_max: Optional[int] = None,
        log_interval: Optional[int] = None,
        eval_interval: Optional[int] = None,
        dataloader_fn: dict = dict(),
        **kwargs,
    ):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.keep_checkpoint_max = keep_checkpoint_max
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.lr_scheduler = lr_scheduler
        self.scheduler_by_epoch = scheduler_by_epoch
        self.visualizer = visualizer
        self.metric = metric

        # build dataloader
        _dataloader_build_fn = (
            default_dataloader_build_fn(**dataloader_fn)
            if isinstance(dataloader_fn, dict)
            else dataloader_fn
        )
        if train_dataset is not None:
            self.train_dataloader = _dataloader_build_fn(train_dataset)
        if val_dataset is not None:
            self.val_dataloader = _dataloader_build_fn(val_dataset)
        self.val_dataset = val_dataset

        self.iters_per_epoch = len(self.train_dataloader)
        self.last_epoch = 0

        # resume
        if resume:
            model_file = os.path.join(self.save_dir, "latest.pdparams")
            print("resume model from {}".format(model_file))
            assert os.path.exists(
                model_file
            ), "Resume model {} not exist.".format(model_file)
            state_dict = paddle.load(model_file)
            self.model.set_state_dict(state_dict["model"])
            self.optimizer.set_state_dict(state_dict["optimizer"])
            self.last_epoch = state_dict["epoch"]

        self.count_model_parameters(self.model)
        self.log_model(self.model)

    def log_model(self, model):
        """log model"""
        logging.info(model.__repr__())

    def count_model_parameters(self, model):
        """count model parameters"""
        logging.info(
            "Overall number of parameters: {}".format(
                sum(p.numel().item() for p in model.parameters())
            )
        )
        if model.with_backbone:
            logging.info(
                "Number of parameters of backbone: {}".format(
                    sum(p.numel().item() for p in model.backbone.parameters())
                )
            )

        if model.with_neck:
            num = sum(p.numel().item() for p in model.neck.parameters())
            logging.info("Number of parameters of neck: {}".format(num))

        if model.with_head:
            num = sum(p.numel().item() for p in model.head.parameters())
            logging.info("Number of parameters of head: {}".format(num))

    def format_msg(self, msg_dict):
        """format msg"""
        msgs = []
        for key, value in msg_dict.items():
            if isinstance(value, float):
                msgs.append("{}={:.6f}".format(key, value))
            elif isinstance(value, paddle.Tensor):
                msgs.append("{}={:.6f}".format(key, value.item()))
            else:
                msgs.append("{}={}".format(key, value))
        msg = ", ".join(msgs)

        return msg

    def parse_losses(self, losses):
        """
        Parse the loss tensor in dictionary into a single scalar.
        """
        log_loss = dict()
        if isinstance(losses, paddle.Tensor):
            total_loss = losses
        elif isinstance(losses, dict):
            for loss_name, loss_value in losses.items():
                if isinstance(loss_value, list):
                    loss_sum = 0
                    for loss_part in loss_value:
                        loss_sum += loss_part
                else:
                    loss_sum = loss_value
                log_loss[loss_name] = loss_sum
            total_loss = 0
            for loss_name, loss_value in log_loss.items():
                if "loss" in loss_name:
                    total_loss += loss_value

        log_loss["total_loss"] = total_loss

        return total_loss, log_loss

    def train(self):
        """train"""
        # sync bn
        sync_bn = (
            getattr(self.model, "sync_bn", False) and dist.get_world_size() > 1
        )
        if sync_bn:
            logging.info("=>>>>>>>>>>>>>>>> Convet bn to sync bn !!!")
            self.model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.model
            )

        if dist.get_world_size() > 1:
            if not dist.is_initialized():
                dist.init_parallel_env()
            self.model = paddle.DataParallel(self.model)

        # Log
        logwriter = LogWriter(logdir=self.save_dir)

        iter = self.last_epoch * self.iters_per_epoch + 1
        iters = self.iters_per_epoch * self.epochs
        timer = Timer(iter, iters)
        for epoch in range(self.last_epoch + 1, self.epochs + 1):
            # Train
            self.model.train()
            if isinstance(self.model, paddle.DataParallel):
                if hasattr(self.model._layers, "convert_to_train"):
                    self.model._layers.convert_to_train()
            elif hasattr(self.model._layers, "convert_to_train"):
                self.model.convert_to_train()
            for sample in self.train_dataloader:
                outputs = self.model(sample)
                total_loss, log_loss = self.parse_losses(outputs["loss"])
                self.optimizer.clear_grad()
                total_loss.backward()
                self.optimizer.step()

                timer.step()
                if dist.get_rank() == 0 and iter % self.log_interval == 0:
                    lr = self.lr_scheduler.get_lr()
                    msg_dict = {
                        "mode": "Train",
                        "epoch": "{}/{}".format(epoch, self.epochs),
                        "iter": "{}/{}".format(iter, iters),
                        "lr": lr,
                        "eta": timer.eta(),
                    }

                    for loss_name, loss_value in log_loss.items():
                        msg_dict[loss_name] = loss_value
                        logwriter.add_scalar(loss_name, loss_value, iter)
                    logwriter.add_scalar("lr", lr, iter)

                    msg = self.format_msg(msg_dict)
                    logging.info(msg)
                iter += 1
                if not self.scheduler_by_epoch:
                    self.lr_scheduler.step()

            if self.scheduler_by_epoch:
                self.lr_scheduler.step()

            # Save model
            if dist.get_rank() == 0 and epoch % self.save_interval == 0:
                state_dict = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch,
                }
                model_file = os.path.join(
                    self.save_dir, "epoch-{:03d}.pdparams".format(epoch)
                )
                latest_model_file = os.path.join(
                    self.save_dir, "latest.pdparams"
                )
                paddle.save(state_dict, model_file)
                if os.path.lexists(latest_model_file):
                    os.remove(latest_model_file)
                os.symlink(os.path.abspath(model_file), latest_model_file)

                # delete old model
                if epoch > self.keep_checkpoint_max:
                    old_model_file = os.path.join(
                        self.save_dir,
                        "epoch-{:03d}.pdparams".format(
                            epoch - self.keep_checkpoint_max
                        ),
                    )
                    logging.info("Pop model: {}".format(old_model_file))
                    if os.path.exists(old_model_file):
                        os.remove(old_model_file)

            # Eval
            if (
                self.eval_interval > 0 and epoch % self.eval_interval == 0
            ) or epoch == self.epochs:
                metric_dict = self.test(do_eval=True)
                if dist.get_rank() == 0:
                    msg_dict = {}
                    for matric_name, matric_value in metric_dict.items():
                        msg_dict[matric_name] = matric_value
                        logwriter.add_scalar(matric_name, matric_value, iter)

                    msg = self.format_msg(msg_dict)
                    logging.info(msg)

        logging.info("Training is complete")

    def test(
        self,
        save_result=False,
        do_eval=False,
        do_visualize=False,
        no_infer=False,
    ):
        """test"""
        if self.val_dataset is None:
            raise RuntimeError(
                "The testing dataset is not specified in the configuration file!"
            )
        elif len(self.val_dataset) == 0:
            raise ValueError(
                "The length of test dataset is 0. Please check if your dataset is valid!"
            )

        if do_visualize and self.visualizer is None:
            raise RuntimeError("Visualizer is not specified!")

        if do_eval and self.metric is None:
            raise RuntimeError("Metric is not specified!")

        logging.info(f"without inference: {no_infer}")
        result_file = os.path.join(self.save_dir, "results.pdparams")
        visualize_dir = os.path.join(self.save_dir, "visualize")
        metric_dir = os.path.join(self.save_dir, "metrics")

        if dist.get_world_size() > 1:
            if not paddle.distributed.is_initialized():
                paddle.distributed.init_parallel_env()
            if not isinstance(self.model, paddle.DataParallel):
                self.model = paddle.DataParallel(self.model)

        if not no_infer:
            # Test
            self.model.eval()
            if isinstance(self.model, paddle.DataParallel):
                if hasattr(self.model._layers, "convert_to_deploy"):
                    self.model._layers.convert_to_deploy()
            elif hasattr(self.model, "convert_to_deploy"):
                self.model.convert_to_deploy()

            part_results = []
            infer_times = []
            with paddle.no_grad():
                for sample in tqdm.tqdm(self.val_dataloader, desc="Test"):
                    start_time = time.time()
                    outputs = self.model(sample)
                    end_time = time.time()
                    infer_times.append(end_time - start_time)
                    results = outputs["pred"]

                    # add img_meta
                    img_metas = sample["img_meta"]
                    for i, result in enumerate(results):
                        result.update(img_metas[i])

                    # add gt
                    if "gt_lines" in sample:
                        gt_lines = sample["gt_lines"]
                        for i, result in enumerate(results):
                            result.update(dict(gt_lines=gt_lines[i]))

                    part_results.extend(results)

            fps = len(self.val_dataloader) / sum(infer_times)
            logging.info("FPS: {:.0f}".format(fps))

            # save results
            if save_result:
                if dist.get_world_size() > 1:
                    results = api_utils.collect_results_cpu(
                        part_results, size=len(self.val_dataset)
                    )
                else:
                    results = part_results
                if dist.get_rank() == 0:
                    logging.info(
                        "Saving eval results at {}".format(result_file)
                    )
                    paddle.save(results, result_file)
        else:
            logging.info("Loading eval results at {}".format(result_file))
            results = paddle.load(result_file)
            part_results = api_utils.split_results_cpu(results)

        # visualize
        if do_visualize:
            self.visualizer.reset()
            self.visualizer.update(part_results)
            self.visualizer.visualize(visualize_dir)

        # evaluate
        metric_dict = dict()
        if do_eval:
            self.metric.reset()
            self.metric.update(part_results)
            metric_dict = self.metric.accumulate(metric_dir)
            logging.info(metric_dict)

        return metric_dict
