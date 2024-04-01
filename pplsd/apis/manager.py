# -*- encoding: utf-8 -*-
"""
@File    :   manager.py
@Time    :   2023/11/05 15:54:14
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import inspect
from collections.abc import Sequence
from typing import Callable, Iterable, Union
import logging


__all__ = [
    "BACKBONES",
    "MODELS",
    "NECKS",
    "HEADS",
    "LOSSES",
    "DATASETS",
    "TRANSFORMS",
    "LR_SCHEDULERS",
    "OPTIMIZERS",
    "LINE_CODERS",
    "MATCH_COSTS",
    "METRICS",
    "VISUALIZERS",
    "POSITIONAL_ENCODINGS",
    "ATTENTIONS",
    "TRANSFORMERS",
    "TRANSFORMER_ENCODERS",
    "TRANSFORMER_DECODERS",
    "TRANSFORMER_ENCODER_LAYERS",
    "TRANSFORMER_DECODER_LAYERS",
]


class ComponentManager:
    """
    Component Manager
    """

    def __init__(self, name: str):
        self._components_dict = dict()
        self._name = name

    def __len__(self):
        return len(self._components_dict)

    def __repr__(self):
        name_str = self._name if self._name else self.__class__.__name__
        return "{}: {}".format(name_str, list(self._components_dict.keys()))

    def __getitem__(self, item: str):
        if item not in self._components_dict.keys():
            raise KeyError(
                "{} does not exist in availabel {}".format(item, self)
            )
        return self._components_dict[item]

    @property
    def components_dict(self) -> dict:
        return self._components_dict

    @property
    def name(self) -> str:
        return self._name

    def _add_single_component(self, component: Callable):
        """
        Add a single component into the corresponding manager.
        Args:
            component (function|class): A new component.
        Raises:
            TypeError: When `component` is neither class nor function.
            KeyError: When `component` was added already.
        """

        # Currently only support class or function type
        if not (inspect.isclass(component) or inspect.isfunction(component)):
            raise TypeError(
                "Expect class/function type, but received {}".format(
                    type(component)
                )
            )

        # Obtain the internal name of the component
        component_name = component.__name__

        # Check whether the component was added already
        if component_name in self._components_dict:
            logging.warn(
                "{} exists already! It is now updated to {} !!!".format(
                    component_name, component
                )
            )
            self._components_dict[component_name] = component

        else:
            # Take the internal name of the component as its key
            self._components_dict[component_name] = component

    def add_component(
        self, components: Union[Callable, Iterable[Callable]]
    ) -> Union[Callable, Iterable[Callable]]:
        """
        Add component(s) into the corresponding manager.
        Args:
            components (function|class|list|tuple): Support four types of components.
        Returns:
            components (function|class|list|tuple): Same with input components.
        """

        # Check whether the type is a sequence
        if isinstance(components, Sequence):
            for component in components:
                self._add_single_component(component)
        else:
            component = components
            self._add_single_component(component)

        return components


MODELS = ComponentManager(name="models")
BACKBONES = ComponentManager(name="backbones")
NECKS = ComponentManager(name="necks")
HEADS = ComponentManager(name="heads")
LOSSES = ComponentManager(name="losses")
DATASETS = ComponentManager(name="datasets")
TRANSFORMS = ComponentManager(name="transforms")
LR_SCHEDULERS = ComponentManager(name="lr_schedulers")
OPTIMIZERS = ComponentManager(name="optimizers")
LINE_CODERS = ComponentManager(name="line_coders")
MATCH_COSTS = ComponentManager(name="match_costs")
METRICS = ComponentManager(name="metrics")
VISUALIZERS = ComponentManager(name="visualizers")
POSITIONAL_ENCODINGS = ComponentManager(name="positional_encodings")
ATTENTIONS = ComponentManager(name="attentions")
TRANSFORMERS = ComponentManager(name="transformers")
TRANSFORMER_ENCODERS = ComponentManager(name="transformer_encoders")
TRANSFORMER_DECODERS = ComponentManager(name="transformer_decoders")
TRANSFORMER_ENCODER_LAYERS = ComponentManager(
    name="transformer_encoder_layers"
)
TRANSFORMER_DECODER_LAYERS = ComponentManager(
    name="transformer_decoder_layers"
)
