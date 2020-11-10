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
        print("inputs:", flush=True)
        print(inputs, flush=True)
        print("gates:", flush=True)
        print(gates, flush=True)
        print("pre hidden:", flush=True)
        print(pre_hidden, flush=True)
        # update reset gate, update gate
        chunked_gates = np.split(gates, 3, -1)

        chunked_weight_hh = np.split(self.weight_hh.T, 3, -1)

        if self.bias_hh is not None:
            chunked_bias_hh = np.split(self.bias_hh, 3, -1)
        for i in range(2):
            chunked_gates[i] += np.matmul(pre_hidden, chunked_weight_hh[i])
            print("chunked_gates {}".format(i), flush=True)
            print(chunked_gates[i], flush=True)
            if self.bias_hh is not None:
                chunked_gates[i] += chunked_bias_hh[i]
        #print("before activation:", flush=True)
        #print("chunked_gates[0]: ", flush=True)
        #print(chunked_gates[0], flush=True)
        #print("chunked_gates[1]: ", flush=True)
        #print(chunked_gates[1], flush=True)

        r = 1.0 / (1.0 + np.exp(-chunked_gates[0]))
        z = 1.0 / (1.0 + np.exp(-chunked_gates[1]))
        print("r gates:", flush=True)
        print(r, flush=True)
        print("z gates:", flush=True)
        print(z, flush=True)
        print("chunked_bias_hh[2]:", flush=True)
        print(chunked_bias_hh[2], flush=True)

        reset_output = np.matmul(pre_hidden, chunked_weight_hh[2])
        print("reset output:", flush=True)
        print(reset_output, flush=True)
        if self.bias_hh is not None:
            reset_output += chunked_bias_hh[2]
        reset_output *= r
        chunked_gates[2] += reset_output
        #print("reset output:", flush=True)
        #print(reset_output, flush=True)
        print("cell_state_value:", flush=True)
        print(chunked_gates[2], flush=True)
        chunked_gates[2] = np.tanh(chunked_gates[2])
        h = (pre_hidden - chunked_gates[2]) * z + chunked_gates[2]
        print("hidden:", flush=True)
        print(h, flush=True)
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
        super(GRU, self).__init__()
        weight_len = len(flat_w)
        if direction in ["forward", "backward"]:
            is_reverse = direction == "backward"
            for i in range(0, num_layers):
                x_size = hidden_size
                if i == 0:
                    x_size = input_size
                cell = GRUCell(x_size, hidden_size, flat_w[i * 2][1],
                               flat_w[i * 2 + 1][1],
                               flat_w[weight_len // 2 + i * 2][1],
                               flat_w[weight_len // 2 + i * 2 + 1][1])
                self.append(RNN(cell, is_reverse, time_major))
        elif direction == "bidirectional":
            for i in range(0, num_layers):
                x_size = hidden_size
                if i == 0:
                    x_size = input_size
                cell_fw = GRUCell(2 * x_size, hidden_size, flat_w[i * 4][1],
                                  flat_w[i * 4 + 1][1],
                                  flat_w[weight_len // 2 + i * 4][1],
                                  flat_w[weight_len // 2 + i * 4 + 1][1])
                cell_bw = GRUCell(2 * x_size, hidden_size, flat_w[i * 4 + 2][1],
                                  flat_w[i * 4 + 3][1],
                                  flat_w[weight_len // 2 + i * 4 + 2][1],
                                  flat_w[weight_len // 2 + i * 4 + 3][1])
                self.append(BiRNN(cell_fw, cell_bw, time_major))
        else:
            raise ValueError(
                "direction should be forward, backward or bidirectional, "
                "received direction = {}".format(direction))

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_directions = 2 if direction == "bidirectional" else 1
        self.time_major = time_major
        self.num_layers = num_layers
        self.state_components = 1


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestGRUOpCpu(OpTest):
    def get_weight_names(self, direction_num):
        weight_names = []
        for i in range(self.num_layers):
            for j in range(0, 2 * direction_num):
                weight_names.append("{}.weigth_{}".format(i, j))
        for i in range(self.num_layers):
            for j in range(0, 2 * direction_num):
                weight_names.append("{}.bias_{}".format(i, j))
        return weight_names

    def setUp(self):
        self.op_type = "cudnn_lstm"
        self.dtype = np.float64
        self.sequence_length = np.array(
            [12, 11, 10, 9, 8, 7, 6, 5], dtype=np.int32)
        self.num_layers = 1
        self.is_bidirec = False
        self.is_test = False
        self.dropout = 0.
        self.set_attrs()

        direction_num = 2 if self.is_bidirec else 1
        direction = "bidirectional" if self.is_bidirec else "forward"
        seq_length = 12
        batch_size = 8
        input_size = 20
        hidden_size = 15
        input = np.random.uniform(
            low=-0.1, high=0.1,
            size=(seq_length, batch_size, input_size)).astype(self.dtype)

        if self.sequence_length is not None:
            input[3][1:][:] = 0
            input[4][2:][:] = 0
            input[2][3:][:] = 0
            input[1][4:][:] = 0

        flat_w = create_parameter_for_rnn(
            input_size,
            hidden_size,
            self.dtype,
            self.num_layers,
            self.is_bidirec,
            gate_num=3)
        #print("flat_w", flush=True)
        #print(flat_w)

        rnn1 = GRU(input_size,
                   hidden_size,
                   num_layers=self.num_layers,
                   time_major=True,
                   direction=direction,
                   dropout=self.dropout,
                   flat_w=flat_w)

        output, last_hidden = rnn1(input, sequence_length=self.sequence_length)

        init_h = np.zeros((self.num_layers * direction_num, batch_size,
                           hidden_size)).astype(self.dtype)

        state_out = np.ndarray((300)).astype("uint8")

        self.inputs = {
            'Input': input,
            'WeightList': flat_w,
            'InitH': init_h,
            'SequenceLength': self.sequence_length
        }
        if self.sequence_length is None:
            self.inputs = {
                'Input': input,
                'WeightList': flat_w,
                'InitH': init_h,
            }
        self.attrs = {
            'dropout_prob': self.dropout,
            'is_bidirec': self.is_bidirec,
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': self.num_layers,
            'is_test': self.is_test,
            'cell_type': "gru"
        }
        self.outputs = {
            'Out': output,
            'LastH': last_hidden,
            'LastC': np.ndarray((400)).astype("uint8"),
            'Reserve': np.ndarray((400)).astype("uint8"),
            'StateOut': state_out
        }

    def set_attrs(self):
        pass

    def test_output_with_place(self):
        place = core.CPUPlace()
        self.check_output_with_place(
            place, no_check_set=['Reserve', 'StateOut', 'LastC'])


#    def test_grad_with_place(self):
#        place = core.CPUPlace()
#        direction_num = 2 if self.is_bidirec else 1
#        var_name_list = self.get_weight_names(direction_num)
#        grad_check_list = []
#        grad_check_list = ['Input', 'InitH']
#        grad_check_list.extend(var_name_list)
#        self.check_grad_with_place(place,
#                                   set(grad_check_list),
#                                   ['Out', 'LastH'])

if __name__ == '__main__':
    unittest.main()
