# -*- encoding: utf-8 -*-
"""
@File    :   config.py
@Time    :   2023/12/16 23:17:17
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import os
from collections.abc import Iterable, Mapping
from typing import Optional
import paddle
from paddle.metric import Metric
import yaml
from lh_pplsd.apis import manager
from lh_pplsd.visualizers import Visualizer


class Config:
    """
    Config
    """

    def __init__(
        self,
        path: str,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        epochs: Optional[int] = None,
    ):
        if not path:
            raise ValueError("Please specify the configuration file path.")

        if not os.path.exists(path):
            raise FileNotFoundError("File {} does not exist".format(path))

        self._model = None
        self._train_dataset = None
        self._val_dataset = None
        if path.endswith("yml") or path.endswith("yaml"):
            self.dic = self._parse_from_yaml(path)
        else:
            raise RuntimeError("Config file should in yaml format!")

        self.update(
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
        )

    def _update_dic(self, dic: dict, base_dic: dict) -> dict:
        """Update config from dic based base_dic"""

        base_dic = base_dic.copy()
        dic = dic.copy()

        if dic.get("_inherited_", True) == False:
            dic.pop("_inherited_")
            return dic

        for key, val in dic.items():
            if isinstance(val, dict) and key in base_dic:
                base_dic[key] = self._update_dic(val, base_dic[key])
            else:
                base_dic[key] = val
        dic = base_dic
        return dic

    def _parse_from_yaml(self, path: str) -> dict:
        """Parse a yaml file and build config"""

        with open(path, "r") as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)

        if "_base_" in dic:
            cfg_dir = os.path.dirname(path)
            base_path = dic.pop("_base_")
            base_path = os.path.join(cfg_dir, base_path)
            base_dic = self._parse_from_yaml(base_path)
            dic = self._update_dic(dic, base_dic)
        return dic

    def update(
        self,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        epochs: Optional[int] = None,
    ):
        """Update config"""

        if learning_rate is not None:
            self.dic["lr_scheduler"]["learning_rate"] = learning_rate

        if batch_size is not None:
            self.dic["batch_size"] = batch_size

        if epochs is not None:
            self.dic["epochs"] = epochs

    @property
    def batch_size(self) -> int:
        return self.dic.get("batch_size", 1)

    @property
    def epochs(self) -> int:
        epochs = self.dic.get("epochs")
        return epochs

    @property
    def lr_scheduler(self) -> paddle.optimizer.lr.LRScheduler:
        if "lr_scheduler" not in self.dic:
            raise RuntimeError(
                "No `lr_scheduler` specified in the configuration file."
            )

        params = self.dic.get("lr_scheduler")
        return self._load_object(params)

    @property
    def scheduler_by_epoch(self) -> int:
        return self.dic.get("scheduler_by_epoch", True)

    @property
    def optimizer(self) -> paddle.optimizer.Optimizer:
        params = self.dic.get("optimizer", {}).copy()

        params["learning_rate"] = self.lr_scheduler
        params["parameters"] = filter(
            lambda p: p.trainable, self.model.parameters()
        )
        optimizer = self._load_object(params)

        return optimizer

    @property
    def visualizer(self) -> Visualizer:
        params = self.dic.get("visualizer", {}).copy()
        if not params:
            return None
        return self._load_object(params)

    @property
    def metric(self) -> Metric:
        params = self.dic.get("metric", {}).copy()
        if not params:
            return None
        return self._load_object(params)

    @property
    def model(self) -> paddle.nn.Layer:
        model_cfg = self.dic.get("model").copy()
        if not model_cfg:
            raise RuntimeError("No model specified in the configuration file.")

        if not self._model:
            self._model = self._load_object(model_cfg)
        return self._model

    @property
    def train_dataset_config(self) -> dict:
        return self.dic.get("train_dataset", {}).copy()

    @property
    def val_dataset_config(self) -> dict:
        return self.dic.get("val_dataset", {}).copy()

    @property
    def train_dataset_class(self):
        dataset_type = self.train_dataset_config["type"]
        return self._load_component(dataset_type)

    @property
    def val_dataset_class(self):
        dataset_type = self.val_dataset_config["type"]
        return self._load_component(dataset_type)

    @property
    def train_dataset(self) -> paddle.io.Dataset:
        _train_dataset = self.train_dataset_config
        if not _train_dataset:
            return None
        if not self._train_dataset:
            self._train_dataset = self._load_object(_train_dataset)
        return self._train_dataset

    @property
    def val_dataset(self) -> paddle.io.Dataset:
        _val_dataset = self.val_dataset_config
        if not _val_dataset:
            return None
        if not self._val_dataset:
            self._val_dataset = self._load_object(_val_dataset)
        return self._val_dataset

    def _load_component(self, com_name: str):
        for com in manager.__all__:
            com = getattr(manager, com)
            if com_name in com.components_dict:
                return com[com_name]
        else:
            if com_name in paddle.optimizer.lr.__all__:
                return getattr(paddle.optimizer.lr, com_name)
            elif com_name in paddle.optimizer.__all__:
                return getattr(paddle.optimizer, com_name)
            elif com_name in paddle.nn.__all__:
                return getattr(paddle.nn, com_name)
            elif com_name in paddle.metric.__all__:
                return getattr(paddle.nn, com_name)
            raise RuntimeError(
                "The specified component was not found {}.".format(com_name)
            )

    def _load_object(self, obj, recursive: bool = True):
        if isinstance(obj, Mapping):
            dic = obj.copy()
            component = (
                self._load_component(dic.pop("type"))
                if "type" in dic
                else dict
            )

            if recursive:
                params = {}
                for key, val in dic.items():
                    params[key] = self._load_object(
                        obj=val, recursive=recursive
                    )
            else:
                params = dic
            try:
                return component(**params)
            except Exception as e:
                raise type(e)("{} {}".format(component.__name__, e))

        elif isinstance(obj, Iterable) and not isinstance(obj, str):
            return [self._load_object(item) for item in obj]

        return obj

    def _is_meta_type(self, item) -> bool:
        return isinstance(item, dict) and "type" in item

    def __str__(self) -> str:
        msg = "---------------Config Information---------------"
        msg += "\n{}".format(yaml.dump(self.dic))
        msg += "------------------------------------------------"
        return msg

    def to_dict(self) -> dict:
        dic = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler,
            "scheduler_by_epoch": self.scheduler_by_epoch,
            "model": self.model,
            "train_dataset": self.train_dataset,
            "val_dataset": self.val_dataset,
            "visualizer": self.visualizer,
            "metric": self.metric,
        }

        return dic
