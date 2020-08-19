# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

import copy
import math
import numpy as np
import unittest

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.framework as framework
import paddle.fluid.core as core


def exponential_decay(learning_rate,
                      global_step,
                      decay_steps,
                      decay_rate,
                      staircase=False):
    exponent = global_step / decay_steps
    if staircase:
        exponent = math.floor(exponent)
    return learning_rate * decay_rate**exponent


def natural_exp_decay(learning_rate,
                      global_step,
                      decay_steps,
                      decay_rate,
                      staircase=False):
    exponent = float(global_step) / float(decay_steps)
    if staircase:
        exponent = math.floor(exponent)
    return learning_rate * math.exp(-1 * decay_rate * exponent)


def inverse_time_decay(learning_rate,
                       global_step,
                       decay_steps,
                       decay_rate,
                       staircase=False):
    temp = float(global_step) / float(decay_steps)
    if staircase:
        temp = math.floor(temp)
    return learning_rate / (1 + decay_rate * temp)


def polynomial_decay(learning_rate,
                     global_step,
                     decay_steps,
                     end_learning_rate=0.0001,
                     power=1.0,
                     cycle=False):
    if cycle:
        div = math.ceil(global_step / float(decay_steps))
        if div == 0:
            div = 1
        decay_steps = decay_steps * div
    else:
        global_step = min(global_step, decay_steps)
    return (learning_rate - end_learning_rate) * \
           ((1 - float(global_step) / float(decay_steps)) ** power) + end_learning_rate


def piecewise_decay(global_step, boundaries, values):
    assert len(boundaries) + 1 == len(values)
    for i in range(len(boundaries)):
        if global_step < boundaries[i]:
            return values[i]
    return values[len(values) - 1]


def cosine_decay(global_step, learning_rate, step_each_epoch, epochs):
    cur_epoch = math.floor(global_step / step_each_epoch)
    decayed_lr = learning_rate * 0.5 * (
        math.cos(cur_epoch * math.pi / epochs) + 1)
    return decayed_lr


def noam_decay(global_step, d_model, warmup_steps, learning_rate=1.0):
    a = math.pow(global_step, -0.5)
    b = math.pow(warmup_steps, -1.5) * global_step
    decayed_lr = learning_rate * math.pow(d_model, -0.5) * min(a, b)

    return decayed_lr


def linear_lr_warmup(global_step, warmup_steps, start_lr, end_lr):
    linear_step = end_lr - start_lr
    decayed_lr = start_lr + linear_step * (global_step / warmup_steps)
    return decayed_lr


def multi_step_decay(global_step, learning_rate, milestones, decay_rate=0.1):
    for i in range(len(milestones)):
        if global_step < milestones[i]:
            return learning_rate * math.pow(decay_rate, i)

    return learning_rate * math.pow(decay_rate, len(milestones))


def step_decay(global_step, learning_rate, step_size, decay_rate=0.1):
    return learning_rate * math.pow(decay_rate, global_step // step_size)


def lambda_decay(global_step, learning_rate, lr_lambda):
    return learning_rate * lr_lambda(global_step)


class TestLearningRateDecayDygraph(unittest.TestCase):
    def test_LR_state_dict(self):
        paddle.disable_static()
        x = np.random.uniform(-1, 1, [3, 10]).astype("float32")
        linear = paddle.nn.Linear(10, 10)
        input = paddle.to_tensor(x)
        Exponential_scheduler = paddle.optimizer.ExponentialLR(
            learning_rate=0.1,
            decay_steps=10000,
            decay_rate=0.5,
            staircase=True)
        Step_scheduler = paddle.optimizer.StepLR(0.5, step_size=3)
        Reducelr_scheduler = paddle.optimizer.ReduceLROnPlateau(
            learning_rate=1.0, decay_rate=0.5, patience=5, cooldown=3)

        adam1 = paddle.optimizer.Adam(
            learning_rate=Exponential_scheduler,
            parameter_list=linear.parameters())
        adam2 = paddle.optimizer.Adam(
            learning_rate=Step_scheduler, parameter_list=linear.parameters())
        adam3 = paddle.optimizer.Adam(
            learning_rate=Reducelr_scheduler,
            parameter_list=linear.parameters())
        print(adam3.state_dict())

        for epoch in range(10):
            out = linear(input)
            loss = paddle.reduce_mean(out)
            loss.backward()
            adam1.minimize(loss)
            adam2.minimize(loss)
            adam3.minimize(loss)
            linear.clear_gradients()

            Step_scheduler.epoch()
            Reducelr_scheduler.step(loss)

        fluid.save_dygraph(linear.state_dict(), "save_path")

        Exponential_scheduler_test = paddle.optimizer.ExponentialLR(
            learning_rate=0.1,
            decay_steps=10000,
            decay_rate=0.5,
            staircase=True)
        Step_scheduler_test = paddle.optimizer.StepLR(0.5, step_size=3)
        Reducelr_scheduler_test = paddle.optimizer.ReduceLROnPlateau(
            learning_rate=1.0, decay_rate=0.5, patience=5, cooldown=3)

        fluid.save_dygraph(adam1.state_dict(), "save_path")
        _, opt_state = fluid.load_dygraph("save_path")
        adam_test = paddle.optimizer.Adam(
            learning_rate=Exponential_scheduler_test,
            parameter_list=linear.parameters())
        adam_test.set_dict(opt_state)
        self.assertEqual(adam_test._learning_rate.step_num,
                         adam1._learning_rate.step_num,
                         "step_num is different before and after set_dict")

        fluid.save_dygraph(adam2.state_dict(), "save_path")
        _, opt_state = fluid.load_dygraph("save_path")
        adam_test = paddle.optimizer.Adam(
            learning_rate=Step_scheduler_test,
            parameter_list=linear.parameters())
        adam_test.set_dict(opt_state)
        self.assertEqual(adam_test._learning_rate.step_num,
                         adam2._learning_rate.step_num,
                         "step_num is different before and after set_dict")
        self.assertEqual(
            adam_test._learning_rate(),
            adam2._learning_rate(),
            "current learning rate is different before and after set_dict")

        fluid.save_dygraph(adam3.state_dict(), "save_path")
        _, opt_state = fluid.load_dygraph("save_path")
        adam_test = fluid.optimizer.Adam(
            learning_rate=Reducelr_scheduler_test,
            parameter_list=linear.parameters())
        adam_test.set_dict(opt_state)
        self.assertEqual(adam_test._learning_rate.best_loss,
                         adam3._learning_rate.best_loss.numpy()[0],
                         "best_loss is different before and after set_dict")
        self.assertEqual(
            adam_test._learning_rate.cooldown_counter,
            adam3._learning_rate.cooldown_counter,
            "cooldown_counter is different before and after set_dict")
        self.assertEqual(
            adam_test._learning_rate.num_bad_epochs,
            adam3._learning_rate.num_bad_epochs,
            "num_bad_epochs is different before and after set_dict")
        self.assertEqual(adam_test._learning_rate.step_num,
                         adam3._learning_rate.step_num,
                         "epoch is different before and after set_dict")
        self.assertEqual(
            adam_test._learning_rate(),
            adam3._learning_rate(),
            "current learning rate is different before and after set_dict")

    def test_NoamDecay(self):
        paddle.disable_static()
        d_model = 0.01
        warmup_steps = 200
        learning_rate = 2.0
        noam_lr = paddle.optimizer.NoamLR(
            d_model=d_model,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate)
        for step in range(5):
            step += 1
            right_result = noam_decay(step, d_model, warmup_steps,
                                      learning_rate)
            fluid_result = noam_lr()

            self.assertAlmostEqual(
                right_result,
                fluid_result[0],
                msg='Failed lr scheduler in step {0}, Python result is {1}, Fluid result is {2}'.
                format(step, right_result, fluid_result[0]))

    def test_LinearLrWarmup(self):
        paddle.disable_static()
        lr = paddle.optimizer.PolynomialLR(
            learning_rate=1.0, decay_steps=10, end_learning_rate=0.0, power=1.0)
        lr = paddle.optimizer.LinearLRWarmup(
            learning_rate=lr, warmup_steps=2, start_lr=0.0, end_lr=1.0)

        right_result = [0.5, 0.9, 0.8, 0.7, 0.6]
        for i in range(5):

            t = lr()

            self.assertTrue(np.allclose((t.numpy())[0].item(), right_result[i]))

        with self.assertRaises(TypeError):
            lr = paddle.optimizer.LinearLRWarmup(
                learning_rate="fake_lr",
                warmup_steps=2,
                start_lr=0.0,
                end_lr=1.0)

    def test_MultiStepDecay(self):
        paddle.disable_static()
        linear = paddle.nn.Linear(10, 10)
        learning_rate = 0.5
        milestones = [2, 4, 8]
        decay_rate = 0.2
        scheduler = paddle.optimizer.MultiStepLR(
            learning_rate=learning_rate,
            milestones=milestones,
            decay_rate=decay_rate)

        adam = paddle.optimizer.AdamOptimizer(
            learning_rate=scheduler, parameter_list=linear.parameters())
        for epoch in range(10):
            right_result = multi_step_decay(epoch, learning_rate, milestones,
                                            decay_rate)
            fluid_result = adam.current_step_lr()
            scheduler.epoch()
            self.assertAlmostEqual(
                right_result,
                fluid_result,
                msg='Failed lr scheduler in epoch {0}, Python result is {1}, Fluid result is {2}'.
                format(epoch, right_result, fluid_result))

        with self.assertRaises(ValueError):
            lr = paddle.optimizer.MultiStepLR(learning_rate, [30, 50, 20], 0.1)

        with self.assertRaises(ValueError):
            lr = paddle.optimizer.MultiStepLR(learning_rate, [20, 30, 50], 1)

        with self.assertRaises(TypeError):
            lr = paddle.optimizer.MultiStepLR("test", [20, 30, 50])

        with self.assertRaises(ValueError):
            lr = paddle.optimizer.MultiStepLR(-1, [20, 30, 50])

    def test_StepDecay(self):
        paddle.disable_static()
        learning_rate = 0.5
        step_size = 3
        decay_rate = 0.2
        scheduler = paddle.optimizer.StepLR(
            learning_rate=learning_rate,
            step_size=step_size,
            decay_rate=decay_rate)
        for epoch in range(10):
            right_result = step_decay(epoch, learning_rate, step_size,
                                      decay_rate)
            fluid_result = scheduler().numpy()[0]
            scheduler.epoch()
            self.assertAlmostEqual(
                right_result,
                fluid_result,
                msg='Failed lr scheduler in epoch {0}, Python result is {1}, Fluid result is {2}'.
                format(epoch, right_result, fluid_result))

        with self.assertRaises(TypeError):
            lr = paddle.optimizer.StepLR(learning_rate, "test", 0.1)

        with self.assertRaises(ValueError):
            lr = paddle.optimizer.StepLR(learning_rate, 20, 2)

    def test_LambdaDecay(self):
        paddle.disable_static()
        learning_rate = 0.5
        lr_lambda = lambda x: 0.95**x
        scheduler = paddle.optimizer.LambdaLR(learning_rate, lr_lambda)

        linear = paddle.nn.Linear(10, 10)
        adam = paddle.optimizer.Adam(
            scheduler, parameter_list=linear.parameters())

        for epoch in range(30):
            right_result = lambda_decay(epoch, learning_rate, lr_lambda)
            fluid_result = scheduler().numpy()[0]
            scheduler.epoch()
            self.assertAlmostEqual(
                right_result,
                fluid_result,
                msg='Failed lr scheduler in epoch {0}, Python result is {1}, Fluid result is {2}'.
                format(epoch, right_result, fluid_result))

        with self.assertRaises(TypeError):
            lr = paddle.optimizer.LambdaLR(learning_rate, "test")


if __name__ == '__main__':
    unittest.main()
