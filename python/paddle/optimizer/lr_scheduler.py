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
    'NoamDecay', 'PiecewiseScheduler', 'NaturalExpScheduler',
    'ExponentialScheduler', 'InverseTimeDecay', 'PolynomialDecay',
    'CosineDecay', 'LinearLrWarmup', 'ReduceLROnPlateau', 'StepDecay',
    'MultiStepDecay', 'LambdaDecay'
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


class PiecewiseScheduler(_LRScheduler):
    """

    Piecewise decay scheduler.

    The algorithm can be described as the code below.

    .. code-block:: text

        boundaries = [10000, 20000]
        values = [1.0, 0.5, 0.1]
        if global_step < 10000:
            learning_rate = 1.0
        elif 10000 <= global_step < 20000:
            learning_rate = 0.5
        else:
            learning_rate = 0.1

    Parameters:
        boundaries(list): A list of steps numbers. The type of element in the list is python int. 
        values(list): A list of learning rate values that will be picked during
            different step boundaries. The type of element in the list is python float.
        begin(int): The begin step to initialize the global_step in the description above.
        step(int, optional): The step size used to calculate the new global_step in the description above.
            The default value is 1.
        dtype(str, optional): The data type used to create the learning rate variable. The data type can be set as
            'float32', 'float64'. The default value is 'float32'.

    Returns:
        None.

    Examples:
        .. code-block:: python
          
          import paddle.fluid as fluid
          boundaries = [10000, 20000]
          values = [1.0, 0.5, 0.1]
          with fluid.dygraph.guard():
              emb = fluid.dygraph.Embedding( [10, 10] )
              optimizer = fluid.optimizer.SGD(
                 learning_rate=fluid.dygraph.PiecewiseDecay(boundaries, values, 0),
                 parameter_list = emb.parameters() )
    """

    # TODO(Jack): modify example

    def __init__(self, boundaries, values, begin, step=1, dtype='float32'):
        super(PiecewiseScheduler, self).__init__(begin, step, dtype)
        self.boundaries = boundaries
        self.values = values

        self.vars = []
        for value in values:
            self.vars.append(value)

    def step(self):
        for i in range(len(self.boundaries)):
            if self.step_num < self.boundaries[i]:
                return self.vars[i]
        return self.create_lr_var(self.vars[len(self.values) - 1])


class NaturalExpScheduler(_LRScheduler):
    """

    Applies natural exponential decay scheduler to the initial learning rate.
    
    The algorithm can be described as following.

    .. math::

        decayed\_learning\_rate = learning\_rate * e^{y} 

    If staircase is set to False, then:

    .. math::

        y = - decay\_rate * \\frac{global\_step}{decay\_steps}

    If staircase is set to True, then:

    .. math::

        y = - decay\_rate * math.floor(\\frac{global\_step}{decay\_steps}) 

    Parameters:
        learning_rate(Variable|float): The initial learning rate. If the type 
            is Variable, it's a tensor with shape [1], the data type can be  
            float32 or float64. It also can be set to python int number.
        decay_steps(int): The decay step size. It determines the decay cycle.
        decay_rate(int): The decay rate.
        staircase(bool, optional): If set to True, decay the learning rate at discrete intervals. The 
            default value is False.
        begin(int, optional): The begin step. The initial value of global_step described above. The default value is 0.
        step(int, optional): The step size used to calculate the new global_step in the description above.
            The default value is 1.
        dtype(str, optional): The data type used to create the learning rate variable. The data type can be set as
            'float32', 'float64'. The default value is 'float32'.

    Returns:
        None.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            base_lr = 0.1
            with fluid.dygraph.guard():
                emb = fluid.dygraph.Embedding([10, 10])
                sgd_optimizer = fluid.optimizer.SGD(
                        learning_rate=fluid.dygraph.NaturalExpDecay(
                            learning_rate=base_lr,
                            decay_steps=10000,
                            decay_rate=0.5,
                            staircase=True),
                        parameter_list=emb.parameters())

    """

    # TODO(Jack): modify example

    def __init__(self,
                 learning_rate,
                 decay_steps,
                 decay_rate,
                 staircase=False,
                 begin=0,
                 step=1,
                 dtype='float32'):
        super(NaturalExpScheduler, self).__init__(begin, step, dtype)
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def step(self):
        from ..fluid import layers
        div_res = self.create_lr_var(self.step_num / self.decay_steps)
        if self.staircase:
            div_res = layers.floor(div_res)
        decayed_lr = self.learning_rate * layers.exp(-1 * self.decay_rate *
                                                     div_res)

        return decayed_lr


class ExponentialScheduler(_LRScheduler):
    """

    Applies exponential decay to the learning rate.

    The algorithm can be described as following.
    
    .. math::

        decayed\_learning\_rate = learning\_rate * decay\_rate ^ y 

    If staircase is set to False, then:

    .. math::

        y = \\frac{global\_step}{decay\_steps} 

    If staircase is set to True, then:

    .. math::

        y = math.floor(\\frac{global\_step}{decay\_steps})


    Parameters:
        learning_rate(Variable|float): The initial learning rate. If the type 
            is Variable, it's a tensor with shape [1], the data type can be  
            float32 or float64. It also can be set to python int number.
        decay_steps(int): The decay step size. It determines the decay cycle.
        decay_rate(float): The decay rate.
        staircase(bool, optional): If set to True, decay the learning rate at discrete intervals. The 
            default value is False.
        begin(int, optional): The begin step. The initial value of global_step described above. The default value is 0.
        step(int, optional): The step size used to calculate the new global_step in the description above.
            The default value is 1.
        dtype(str, optional): The data type used to create the learning rate variable. The data type can be set as
            'float32', 'float64'. The default value is 'float32'.

    Returns:
        None.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          base_lr = 0.1
          with fluid.dygraph.guard():
              sgd_optimizer = fluid.optimizer.SGD(
    	            learning_rate=fluid.dygraph.ExponentialDecay(
		        learning_rate=base_lr,
    		        decay_steps=10000,
		        decay_rate=0.5,
		        staircase=True))

    """

    # TODO(Jack): modify example

    def __init__(self,
                 learning_rate,
                 decay_steps,
                 decay_rate,
                 staircase=False,
                 begin=0,
                 step=1,
                 dtype='float32'):
        super(ExponentialScheduler, self).__init__(begin, step, dtype)
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def step(self):
        from .. import layers
        div_res = self.create_lr_var(self.step_num / self.decay_steps)
        if self.staircase:
            div_res = layers.floor(div_res)

        decayed_lr = self.learning_rate * (self.decay_rate**div_res)

        return decayed_lr
