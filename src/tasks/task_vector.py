import abc
import logging
import math
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
from torch import Tensor, nn

from ..type import StateDict

log = logging.getLogger(__name__)


class TaskVector:
    def __init__(
        self,
        pretrained_checkpoint: str = None,
        finetuned_checkpoint: str = None,
        vector: StateDict = None,
    ):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.

        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.

        Args:
            pretrained_checkpoint (str, optional): Path to the pretrained checkpoint file. Defaults to None.
            finetuned_checkpoint (str, optional): Path to the finetuned checkpoint file. Defaults to None.
            vector (STATE_DICT, optional): The task vector state dict. Defaults to None.

        Raises:
            ValueError: If both `pretrained_checkpoint` and `vector` are None.
            ValueError: If both `finetuned_checkpoint` and `vector` are None.
            ValueError: If both `pretrained_checkpoint` and `finetuned_checkpoint` are None.
            ValueError: If `pretrained_checkpoint` or `finetuned_checkpoint` is not a valid file path.
        """
        if isinstance(pretrained_checkpoint, Path):
            pretrained_checkpoint = str(pretrained_checkpoint)
        if isinstance(finetuned_checkpoint, Path):
            finetuned_checkpoint = str(finetuned_checkpoint)

        if vector is not None:
            self.vector = vector
        else:
            # construct the task vector from the pretrained and finetuned checkpoints
            assert (
                pretrained_checkpoint is not None and finetuned_checkpoint is not None
            )
            with torch.no_grad():
                log.info("TaskVector: " + finetuned_checkpoint)
                pretrained_state_dict = torch.load(
                    pretrained_checkpoint, map_location="cpu"
                ).state_dict()
                finetuned_state_dict = torch.load(
                    finetuned_checkpoint, map_location="cpu"
                ).state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = (
                        finetuned_state_dict[key] - pretrained_state_dict[key]
                    )

    def __add__(self, other: "TaskVector"):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other: "TaskVector"):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = -self.vector[key]
        return TaskVector(vector=new_vector)

    def weightmerging(self, taskvectors, coefficients):
        with torch.no_grad():
            new_vector = {}
            for key in taskvectors[0].vector:
                new_vector[key] = sum(
                    coefficients[k] * taskvectors[k][key]
                    for k in range(len(taskvectors))
                )
        return TaskVector(vector=new_vector)

    def apply_to(
        self, pretrained_checkpoint: str, scaling_coef: float = 1.0
    ) -> nn.Module:
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model: nn.Module = torch.load(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(
                        f"Warning: key {key} is present in the pretrained state dict but not in the task vector"
                    )
                    continue
                new_state_dict[key] = (
                    pretrained_state_dict[key] + scaling_coef * self.vector[key]
                )
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model


class NonLinearTaskVector:
    """
    A class representing a task vector, which is the difference between the parameters of a finetuned model and a pretrained model.

    Attributes:
        task_vector (Dict[str, Tensor]): A dictionary of tensors representing the task vector.
    """

    def __init__(
        self,
        pretrained_model: nn.Module = None,
        finetuned_model: nn.Module = None,
        task_vector: Dict[str, Tensor] = None,
    ):
        """
        Initializes a TaskVector object.

        Args:
            pretrained_model (nn.Module, optional): A pretrained model. Defaults to None.
            finetuned_model (nn.Module, optional): A finetuned model. Defaults to None.
            task_vector (Dict[str, Tensor], optional): A dictionary of tensors representing the task vector. Defaults to None.

        Raises:
            AssertionError: Raised if both models and task vector are provided or if neither is provided.
        """

        model_provided = (pretrained_model is not None) and (
            finetuned_model is not None
        )
        task_vector_provided = task_vector is not None
        # check that either models or task vector is provided, but not both
        assert (
            model_provided or task_vector_provided
        ), "Either models or task vector must be provided."
        assert not (
            model_provided and task_vector_provided
        ), "Either models or task vector must be provided, but not both."

        if task_vector_provided:  # task vector provided
            self.task_vector = task_vector
        else:  # models provided
            with torch.no_grad():
                pretrained_params_dict = pretrained_model.state_dict()
                finetuned_state_dict = finetuned_model.state_dict()
                task_vector = {}
                for k in pretrained_params_dict:
                    task_vector[k] = finetuned_state_dict[k] - pretrained_params_dict[k]
            self.task_vector = task_vector

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = -self.vector[key]
        return self.__class__(task_vector=new_vector)

    def __add__(self, other: "NonLinearTaskVector"):
        """Add two task vectors together."""
        assert isinstance(other, NonLinearTaskVector)
        with torch.no_grad():
            new_vector = {}
            for key in self.task_vector:
                if key not in other.task_vector:
                    log.warn(f"Warning, key {key} is not present in both task vectors.")
                    continue
                new_vector[key] = self.task_vector[key] + other.task_vector[key]
        return self.__class__(task_vector=new_vector)

    def __sub__(self, other):
        """Subtract two task vectors."""
        return self.__add__(-other)

    def __pow__(self, power: float):
        """Power of a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.task_vector:
                new_vector[key] = self.task_vector[key] ** power
        return self.__class__(task_vector=new_vector)

    def __mul__(self, other: float):
        """Multiply a task vector by a scalar."""
        with torch.no_grad():
            new_vector = {}
            for key in self.task_vector:
                new_vector[key] = other * self.task_vector[key]
        return self.__class__(task_vector=new_vector)

    def dot(self, other: "NonLinearTaskVector"):
        """Dot product of two task vectors."""
        assert isinstance(other, NonLinearTaskVector)
        with torch.no_grad():
            dot_product = 0.0
            for key in self.task_vector:
                if key not in other.task_vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                dot_product += torch.sum(
                    self.task_vector[key] * other.task_vector[key]
                ).item()
        return dot_product

    def norm(self):
        """Norm of a task vector."""
        return math.sqrt(self.dot(self))

    def apply_to(
        self,
        pretrained_model: nn.Module,
        scaling_coef: float = 1.0,
    ) -> nn.Module:
        """
                Applies a task vector to a pretrained model, in-place.

                Args:
                    pretrained_model (nn.Module): The pretrained model to apply the task vector to.
                    scaling_coef (float, optional): A scaling factor to apply to the task vector. Defaults to 1.0.
        d
                Returns:
                    nn.Module: The modified pretrained model.

                Raises:
                    None.

                The method modifies the `pretrained_model` in-place by adding the `scaling_coef` times the `task_vector` to its state dictionary.
                The `task_vector` is a dictionary that maps keys to tensors, where each key corresponds to a parameter in the model.
                If a key is present in the `pretrained_model` but not in the `task_vector`, a warning is logged and the key is skipped.
        """
        with torch.no_grad():
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.task_vector:
                    log.warn(
                        f"Warning: key {key} is present in the pretrained state dict but not in the task vector"
                    )  # noqa: E501
                    continue
                new_state_dict[key] = (
                    pretrained_state_dict[key] + scaling_coef * self.task_vector[key]
                )
        pretrained_model.load_state_dict(new_state_dict)
        return pretrained_model
