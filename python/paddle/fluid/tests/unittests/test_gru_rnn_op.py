#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from test_lstm_cudnn_op import LSTMCell, RNNMixin
from test_lstm_cudnn_op import create_parameter_for_rnn, RNN, BiRNN

import unittest
import numpy as np
import math

from op_test import OpTest
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import random
random.seed(2)
np.set_printoptions(threshold=np.inf)
paddle.enable_static()


class GRUCell(LSTMCell):
    def __init__(self,
                 input_size,
                 hidden_size,
                 weight_ih,
                 weight_hh,
                 bias_ih,
                 bias_hh,
                 bias=True):
        super(GRUCell, self).__init__(input_size, hidden_size, weight_ih,
                                      weight_hh, bias_ih, bias_hh, bias)

    def init_state(self, inputs):
        batch_size = inputs.shape[0]
        init_h = np.zeros((batch_size, self.hidden_size), dtype=inputs.dtype)
        return init_h

    def forward(self, inputs, pre_hidden=None):
        if pre_hidden is None:
            pre_hidden = self.init_state(inputs)
        gates = np.matmul(inputs, self.weight_ih.T)
        if self.bias_ih is not None:
            gates += self.bias_ih
        # update reset gate, update gate
        gates[:2] += np.matmul(pre_hidden, self.weight_hh[:2].T)
        if self.bias_hh is not None:
            gates[:2] += self.bias_hh[:2]
        chunked_gates = np.split(gates, 3, -1)
        r = 1.0 / (1.0 + np.exp(-chunked_gates[0]))
        z = 1.0 / (1.0 + np.exp(-chunked_gates[1]))
        # update cell
        reset_output = np.matmul(pre_hidden, self.weight_hh[-1].T)
        reset_output += self.bias_hh[-1]
        reset_output *= r
        chunked_gates[2] += reset_output
        chunked_gates[2] = np.tanh(chunked_gates[2])

        h = z * pre_hidden + (1 - z) * chunked_gates[2]
        return h, h


class GRU(RNNMixin):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 direction="forward",
                 dropout=0.,
                 time_major=False,
                 flat_w=None):
        pass
