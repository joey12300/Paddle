# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import math
import warnings

from .. import unique_name
from ..fluid.framework import Variable
from ..fluid.data_feeder import check_type

__all__ = [
    'NoamDecay', 'PiecewiseDecay', 'NaturalExpDecay', 'ExponentialDecay',
    'InverseTimeDecay', 'PolynomialDecay', 'CosineDecay', 'LinearLrWarmup',
    'ReduceLROnPlateau', 'StepDecay', 'MultiStepDecay', 'LambdaDecay'
]


class _LRScheduler(object):
    """
    Base class of learning rate scheduler
    
    Define the common interface of an _LRScheduler.
    User should not use this class directly,
    but need to use one of it's implementation.
    """

    def __init__(self, begin=0, step=1, dtype='float32'):
        self.step_num = begin
        self.step_size = step
        self.dtype = dtype

    def __call__(self):
        if isinstance(lr, float):
            lr = self.create_lr_var(lr)
        self.step_num += self.step_size
        return lr

    def create_lr_var(self, lr):
        """
        convert lr from float to variable

        Args: 
            lr: learning rate
        Returns:
            learning rate variable
        """
        from ..fluid import layers
        lr = layers.create_global_var(
            name=unique_name.generate("learning_rate"),
            shape=[1],
            value=float(lr),
            dtype=self.dtype,
            persistable=False)
        return lr

    def state_dict(self):
        """
        Returns the state of the scheduler as a :class:`dict`.

        It is a subset of self.__dict__ .
        """
        self._state_keys()
        state_dict = {}
        for key in self.keys:
            if key not in self.__dict__:
                continue
            value = self.__dict__[key]
            if isinstance(value, Variable):
                assert value.shape == [
                    1
                ], "shape of Variable in state_dict must be [1] {}".format(
                    value.shape)
                value = value.numpy()[0]
            state_dict[key] = value

        return state_dict

    def _state_keys(self):
        """
        set the keys in self.__dict__ that are needed to be saved.
        """
        self.keys = ['step_num']

    def load_state_dict(self, state_dict):
        """
        Loads the schedulers state.
        """
        self._state_keys()
        for key in self.keys:
            if key in state_dict:
                self.__dict__[key] = state_dict[key]
            else:
                raise RuntimeError(
                    "Please check whether state_dict is correct for optimizer. Can't find [ {} ] in state_dict".
                    format(key))
        if len(state_dict) > len(self.keys):
            warnings.warn(
                "There are some unused values in state_dict. Maybe the optimizer have different 'LearningRateDecay' when invoking state_dict and set_dict"
            )

    def step(self):
        raise NotImplementedError()
