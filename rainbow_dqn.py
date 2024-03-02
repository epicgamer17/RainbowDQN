import os
# os.environ["OMP_NUM_THREADS"] = f"{1}"
# os.environ['TF_NUM_INTEROP_THREADS'] = f"{1}"
# os.environ['TF_NUM_INTRAOP_THREADS'] = f"{1}"

import tensorflow as tf
# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

import datetime
import numpy as np
import copy
import matplotlib.pyplot as plt
from IPython.display import clear_output
# import search
from collections import deque
from typing import Deque, Dict, List, Tuple
import gymnasium as gym
from time import time
# import moviepy

# from segment_tree import MinSegmentTree, SumSegmentTree
# -*- coding: utf-8 -*-
"""Segment tree for Prioritized Replay Buffer."""

import operator
from typing import Callable


class SegmentTree:
    """ Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    Attributes:
        capacity (int)
        tree (list)
        operation (function)

    """

    def __init__(self, capacity: int, operation: Callable, init_value: float):
        """Initialization.

        Args:
            capacity (int)
            operation (function)
            init_value (float)

        """
        assert (
            capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power of 2."
        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation

    def _operate_helper(
        self, start: int, end: int, node: int, node_start: int, node_end: int
    ) -> float:
        """Returns result of operation in segment."""
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._operate_helper(start, mid, 2 * node, node_start, mid),
                    self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )

    def operate(self, start: int = 0, end: int = 0) -> float:
        """Returns result of applying `self.operation`."""
        if end <= 0:
            end += self.capacity
        end -= 1

        return self._operate_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float):
        """Set value in tree."""
        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        """Get real value in leaf node of tree."""
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]


class SumSegmentTree(SegmentTree):
    """ Create SumSegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(self, capacity: int):
        """Initialization.

        Args:
            capacity (int)

        """
        super(SumSegmentTree, self).__init__(
            capacity=capacity, operation=operator.add, init_value=0.0
        )

    def sum(self, start: int = 0, end: int = 0) -> float:
        """Returns arr[start] + ... + arr[end]."""
        return super(SumSegmentTree, self).operate(start, end)

    def retrieve(self, upperbound: float) -> int:
        """Find the highest index `i` about upper bound in the tree"""
        # TODO: Check assert case and fix bug
        assert 0 <= upperbound <= self.sum() + 1e-5, "upperbound: {}".format(upperbound)

        idx = 1

        while idx < self.capacity:  # while non-leaf
            left = 2 * idx
            right = left + 1
            if self.tree[left] > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.tree[left]
                idx = right
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    """ Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(self, capacity: int):
        """Initialization.

        Args:
            capacity (int)

        """
        super(MinSegmentTree, self).__init__(
            capacity=capacity, operation=min, init_value=float("inf")
        )

    def min(self, start: int = 0, end: int = 0) -> float:
        """Returns min(arr[start], ...,  arr[end])."""
        return super(MinSegmentTree, self).operate(start, end)
class ReplayBuffer:
    def __init__(self, observation_dimensions, max_size: int, batch_size = 32, n_step = 1, gamma = 0.99):
        # self.observation_buffer = np.zeros((max_size,) + observation_dimensions, dtype=np.float32)
        # self.next_observation_buffer = np.zeros((max_size,) + observation_dimensions, dtype=np.float32)
        observation_buffer_shape = []
        observation_buffer_shape += [max_size]
        observation_buffer_shape += list(observation_dimensions)
        observation_buffer_shape = list(observation_buffer_shape)
        self.observation_buffer = np.zeros(observation_buffer_shape, dtype=np.float32)
        self.next_observation_buffer = np.zeros(observation_buffer_shape, dtype=np.float32)
        self.action_buffer = np.zeros(max_size, dtype=np.int32)
        self.reward_buffer = np.zeros(max_size, dtype=np.float32)
        self.done_buffer = np.zeros(max_size)

        self.max_size = max_size
        self.batch_size = batch_size
        self.pointer = 0
        self.size = 0

        # n-step learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(self, observation, action, reward, next_observation, done):
        # print("Storing in Buffer")
        # time1 = 0
        # time1 = time()
        transition = (observation, action, reward, next_observation, done)
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.n_step:
            # print("Buffer Storage Time ", time() - time1)
            return ()

        # compute n-step return and store
        reward, next_observation, done = self._get_n_step_info()
        observation, action = self.n_step_buffer[0][:2]
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.next_observation_buffer[self.pointer] = next_observation
        self.done_buffer[self.pointer] = done

        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        # print("Buffer Storage Time ", time() - time1)
        return self.n_step_buffer[0]

    def sample(self):
        # print("Sampling From Buffer")
        # time1 = time()
        idx = np.random.choice(self.size, self.batch_size, replace=False)

        # print("Buffer Sampling Time ", time() - time1)
        return dict(
            observations=self.observation_buffer[idx],
            next_observations=self.next_observation_buffer[idx],
            actions=self.action_buffer[idx],
            rewards=self.reward_buffer[idx],
            dones=self.done_buffer[idx],
        )

    def sample_from_indices(self, indices):
        # print("Sampling From Indices")
        return dict(
            observations=self.observation_buffer[indices],
            next_observations=self.next_observation_buffer[indices],
            actions=self.action_buffer[indices],
            rewards=self.reward_buffer[indices],
            dones=self.done_buffer[indices],
        )

    def _get_n_step_info(self):
        reward, next_observation, done = self.n_step_buffer[-1][-3:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]
            reward = r + self.gamma * reward * (1 - d)
            next_observation, done = (n_o, d) if d else (next_observation, done)

        return reward, next_observation, done

    def __len__(self):
        return self.size
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
            self,
            observation_dimensions,
            max_size,
            batch_size=32,
            max_priority=1.0,
            alpha=0.6,
            # epsilon=0.01,
            n_step=1,
            gamma=0.99,
        ):
        assert alpha >= 0

        super(PrioritizedReplayBuffer, self).__init__(
            observation_dimensions, max_size, batch_size, n_step=n_step, gamma=gamma
        )

        self.max_priority = max_priority  # (initial) priority
        self.tree_pointer = 0

        self.alpha = alpha  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
        # self.epsilon = epsilon
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(self, observation, action, reward, next_observation, done):
        # print("Storing in PrioritizedReplayBuffer")
        # time1 = 0
        # time1 = time()
        transition = super().store(observation, action, reward, next_observation, done)

        if transition:
            self.sum_tree[self.tree_pointer] = self.max_priority ** self.alpha
            self.min_tree[self.tree_pointer] = self.max_priority ** self.alpha
            self.tree_pointer = (self.tree_pointer + 1) % self.max_size

        # print("Storing in PrioritizedReplayBuffer Time ", time() - time1)
        return transition

    def sample(self, beta=0.4):
        # print("Sampling from PrioritizedReplayBuffer")
        # time1 = 0
        # time1 = time()
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()
        # print("Retrieving Data from PrioritizedReplayBuffer Data Arrays")
        # time2 = 0
        # time2 = time()
        observations = self.observation_buffer[indices]
        next_observations = self.next_observation_buffer[indices]
        actions = self.action_buffer[indices]
        rewards = self.reward_buffer[indices]
        dones = self.done_buffer[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        # print("Retrieving Data from PrioritizedReplayBuffer Data Arrays Time ", time() - time2)

        # print("Sampling from PrioritizedReplayBuffer Time ", time() - time1)
        return dict(
            observations=observations,
            next_observations=next_observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices, priorities):
        assert len(indices) == len(priorities)
        # priorities += self.self.epsilon
        for index, priority in zip(indices, priorities):
            # print("Priority", priority)
            assert priority > 0, ("Negative priority: {}".format(priority))
            assert 0 <= index < len(self)

            self.sum_tree[index] = priority ** self.alpha
            self.min_tree[index] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority) # could remove and clip priorities in experience replay isntead

    def _sample_proportional(self):
        # print("Getting Indices from PrioritizedReplayBuffer Sum Tree")
        # time1 = 0
        # time1 = time()
        indices = []
        total_priority = self.sum_tree.sum(0, len(self) - 1)
        priority_segment = total_priority / self.batch_size

        for i in range(self.batch_size):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            upperbound = np.random.uniform(a, b)
            index = self.sum_tree.retrieve(upperbound)
            indices.append(index)

        # print("Getting Indices from PrioritizedReplayBuffer Sum Tree Time ", time() - time1)
        return indices

    def _calculate_weight(self, index, beta):
        min_priority = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (min_priority * len(self)) ** (-beta)
        priority_sample = self.sum_tree[index] / self.sum_tree.sum()
        weight = (priority_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight
class FastSumTree(object):
    # https://medium.com/free-code-camp/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682

    def __init__(self, capacity):
        self.capacity = int(
            capacity
        )  # number of leaf nodes (final nodes) that contains experiences

        self.tree = np.zeros(2 * self.capacity - 1)  # sub tree
        # self.data = np.zeros(self.capacity, object)  # contains the experiences

    def add(self, idx: int, val: float):
        """Set value in tree."""
        tree_index = idx + self.capacity - 1
        # self.data[self.data_pointer] = data
        self.update(tree_index, val)

    def __getitem__(self, idx: int) -> float:
        """Get real value in leaf node of tree."""
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]

    def update(self, tree_index, val):
        change = val - self.tree[tree_index]
        # print("change", change)
        self.tree[tree_index] = val
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
            # print("new value", self.tree[tree_index])


    def retrieve(self, v):
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        return leaf_index, self.tree[leaf_index]

    @property
    def total_priority(self):
        return self.tree[0]
class FastPrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
            self,
            observation_dimensions,
            max_size,
            batch_size=32,
            max_priority=1.0,
            alpha=0.6,
            # epsilon=0.01,
            n_step=1,
            gamma=0.99,
        ):
        assert alpha >= 0

        super(FastPrioritizedReplayBuffer, self).__init__(
            observation_dimensions, max_size, batch_size, n_step=n_step, gamma=gamma
        )

        self.max_priority = max_priority  # (initial) priority
        self.min_priority = max_priority
        self.tree_pointer = 0

        self.alpha = alpha  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
        # self.epsilon = epsilon

        self.tree = FastSumTree(self.max_size)

    def store(self, observation, action, reward, next_observation, done):
        # print("Storing in PrioritizedReplayBuffer")
        # time1 = 0
        # time1 = time()
        transition = super().store(observation, action, reward, next_observation, done)

        # max_priority = np.max(self.tree.tree[-self.tree.capacity :])
        # if max_priority == 0:
        #     max_priority = self.max_priority

        if transition:
            self.tree.add(self.tree_pointer, self.max_priority)
            self.tree_pointer = (self.tree_pointer + 1) % self.max_size

        # print("Storing in PrioritizedReplayBuffer Time ", time() - time1)
        return transition

    def sample(self, beta=0.4):
        # print("Sampling from PrioritizedReplayBuffer")
        # time1 = 0
        # time1 = time()
        assert len(self) >= self.batch_size
        assert beta > 0

        # indices = self._sample_proportional()
        # print("Getting Indices from PrioritizedReplayBuffer Sum Tree")
        # time1 = 0
        # time1 = time()
        priority_segment = self.tree.total_priority / self.batch_size
        indices, weights = np.empty((self.batch_size,), dtype=np.int32), np.empty(
            (self.batch_size, 1), dtype=np.float32
        )
        # print("Total Priority",self.tree.total_priority)
        for i in range(self.batch_size):
            a, b = priority_segment * i, priority_segment * (i + 1)
            # print(a, b)
            # print("a, b", a, b)
            value = np.random.uniform(a, b)
            index, priority = self.tree.retrieve(value)
            sampling_probabilities = priority / self.tree.total_priority
            # weights[i, 0] = (self.batch_size * sampling_probabilities) ** -beta
            weights[i, 0] = (len(self) * sampling_probabilities) ** -beta
            indices[i] = index - self.tree.capacity + 1
            indices[i] = index - self.tree.capacity + 1

        # max_weight = max(weights)
        max_weight = (len(self) * self.min_priority / self.tree.total_priority) ** -beta
        weights = weights / max_weight

        # print(weights)
        # print("Getting Indices from PrioritizedReplayBuffer Sum Tree Time ", time() - time1)
        # print("Retrieving Data from PrioritizedReplayBuffer Data Arrays")
        # time2 = 0
        # time2 = time()
        observations = self.observation_buffer[indices]
        next_observations = self.next_observation_buffer[indices]
        actions = self.action_buffer[indices]
        rewards = self.reward_buffer[indices]
        dones = self.done_buffer[indices]
        # weights = np.array([self._calculate_weight(i, beta) for i in indices])
        # print("Retrieving Data from PrioritizedReplayBuffer Data Arrays Time ", time() - time2)

        # print("Sampling from PrioritizedReplayBuffer Time ", time() - time1)
        return dict(
            observations=observations,
            next_observations=next_observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices, priorities):
        assert len(indices) == len(priorities)
        # priorities += self.epsilon

        for index, priority in zip(indices, priorities):
            assert priority > 0, "Negative priority: {}".format(priority)
            # assert 0 <= index < len(self)
            # self.tree[index] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority ** self.alpha)
            self.min_priority = min(self.min_priority, priority ** self.alpha)
            # priority = np.clip(priority, self.epsilon, self.max_priority)
            self.tree.update(index + self.tree.capacity - 1, priority ** self.alpha)

# From tensorflow_addons
import tensorflow as tf
from tensorflow.keras import (
    activations,
    initializers,
    regularizers,
    constraints,
)
from tensorflow.keras import backend as K
from tensorflow.keras.layers import InputSpec

def _scaled_noise(size, dtype):
    x = tf.random.normal(shape=size, dtype=dtype)
    return tf.sign(x) * tf.sqrt(tf.abs(x))

class NoisyDense(tf.keras.layers.Dense):
    def __init__(
        self,
        units: int,
        sigma: float = 0.5, # might want to make sigma 0.1 for CPU's
        use_factorised: bool = True,
        activation = None,
        use_bias: bool = True,
        kernel_regularizer = None,
        bias_regularizer = None,
        activity_regularizer = None,
        kernel_constraint = None,
        bias_constraint = None,
        **kwargs,
    ):
        super().__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )
        delattr(self, "kernel_initializer")
        delattr(self, "bias_initializer")
        self.sigma = sigma
        self.use_factorised = use_factorised

    def build(self, input_shape):
        # Make sure dtype is correct
        dtype = tf.dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                "Unable to build `Dense` layer with non-floating point "
                "dtype %s" % (dtype,)
            )

        input_shape = tf.TensorShape(input_shape)
        self.last_dim = tf.compat.dimension_value(input_shape[-1])
        sqrt_dim = self.last_dim ** (1 / 2)
        if self.last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to `Dense` "
                "should be defined. Found `None`."
            )
        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.last_dim})

        # use factorising Gaussian variables
        if self.use_factorised:
            mu_init = 1.0 / sqrt_dim
            sigma_init = self.sigma / sqrt_dim
        # use independent Gaussian variables
        else:
            mu_init = (3.0 / self.last_dim) ** (1 / 2)
            sigma_init = 0.017

        sigma_init = initializers.Constant(value=sigma_init)
        mu_init = initializers.RandomUniform(minval=-mu_init, maxval=mu_init)

        # Learnable parameters
        self.sigma_kernel = self.add_weight(
            "sigma_kernel",
            shape=[self.last_dim, self.units],
            initializer=sigma_init,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )

        self.mu_kernel = self.add_weight(
            "mu_kernel",
            shape=[self.last_dim, self.units],
            initializer=mu_init,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )

        self.eps_kernel = self.add_weight(
            "eps_kernel",
            shape=[self.last_dim, self.units],
            initializer=initializers.Zeros(),
            regularizer=None,
            constraint=None,
            dtype=self.dtype,
            trainable=False,
        )

        if self.use_bias:
            self.sigma_bias = self.add_weight(
                "sigma_bias",
                shape=[
                    self.units,
                ],
                initializer=sigma_init,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )

            self.mu_bias = self.add_weight(
                "mu_bias",
                shape=[
                    self.units,
                ],
                initializer=mu_init,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )

            self.eps_bias = self.add_weight(
                "eps_bias",
                shape=[
                    self.units,
                ],
                initializer=initializers.Zeros(),
                regularizer=None,
                constraint=None,
                dtype=self.dtype,
                trainable=False,
            )
        else:
            self.sigma_bias = None
            self.mu_bias = None
            self.eps_bias = None
        self.reset_noise()
        self.built = True

    @property
    def kernel(self):
        return self.mu_kernel + (self.sigma_kernel * self.eps_kernel)

    @property
    def bias(self):
        if self.use_bias:
            return self.mu_bias + (self.sigma_bias * self.eps_bias)

    def reset_noise(self):
        """Create the factorised Gaussian noise."""

        if self.use_factorised:
            # Generate random noise
            in_eps = _scaled_noise([self.last_dim, 1], dtype=self.dtype)
            out_eps = _scaled_noise([1, self.units], dtype=self.dtype)

            # Scale the random noise
            self.eps_kernel.assign(tf.matmul(in_eps, out_eps))
            self.eps_bias.assign(out_eps[0])
        else:
            # generate independent variables
            self.eps_kernel.assign(
                tf.random.normal(shape=[self.last_dim, self.units], dtype=self.dtype)
            )
            self.eps_bias.assign(
                tf.random.normal(
                    shape=[
                        self.units,
                    ],
                    dtype=self.dtype,
                )
            )

    def remove_noise(self):
        """Remove the factorised Gaussian noise."""

        self.eps_kernel.assign(tf.zeros([self.last_dim, self.units], dtype=self.dtype))
        self.eps_bias.assign(tf.zeros([self.units], dtype=self.dtype))

    def call(self, inputs):
        # TODO(WindQAQ): Replace this with `dense()` once public.
        return super().call(inputs)

    def get_config(self):
        # TODO(WindQAQ): Get rid of this hacky way.
        config = super(tf.keras.layers.Dense, self).get_config()
        config.update(
            {
                "units": self.units,
                "sigma": self.sigma,
                "use_factorised": self.use_factorised,
                "activation": activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "bias_regularizer": regularizers.serialize(self.bias_regularizer),
                "activity_regularizer": regularizers.serialize(
                    self.activity_regularizer
                ),
                "kernel_constraint": constraints.serialize(self.kernel_constraint),
                "bias_constraint": constraints.serialize(self.bias_constraint),
            }
        )
        return config
class Network(tf.keras.Model):
    def __init__(self, config, output_size, input_shape, *args, **kwargs):
        super().__init__()
        self.config = config
        kernel_initializers = []
        for i in range(len(config['conv_layers']) + config['dense_layers'] + config['value_hidden_layers'] + config['advantage_hidden_layers'] + 2):
            if config['kernel_initializer'] == 'glorot_uniform':
                kernel_initializers.append(initializers.glorot_uniform(seed=np.random.seed()))
            elif config['kernel_initializer'] == 'glorot_normal':
                kernel_initializers.append(initializers.glorot_normal(seed=np.random.seed()))
            elif config['kernel_initializer'] == 'he_normal':
                kernel_initializers.append(initializers.he_normal(seed=np.random.seed()))
            elif config['kernel_initializer'] == 'he_uniform':
                kernel_initializers.append(initializers.he_uniform(seed=np.random.seed()))
            elif config['kernel_initializer'] == 'variance_baseline':
                kernel_initializers.append(initializers.VarianceScaling(seed=np.random.seed()))
            elif config['kernel_initializer'] == 'variance_0.1':
                kernel_initializers.append(initializers.VarianceScaling(scale=0.1, seed=np.random.seed()))
            elif config['kernel_initializer'] == 'variance_0.3':
                kernel_initializers.append(initializers.VarianceScaling(scale=0.3, seed=np.random.seed()))
            elif config['kernel_initializer'] == 'variance_0.8':
                kernel_initializers.append(initializers.VarianceScaling(scale=0.8, seed=np.random.seed()))
            elif config['kernel_initializer'] == 'variance_3':
                kernel_initializers.append(initializers.VarianceScaling(scale=3, seed=np.random.seed()))
            elif config['kernel_initializer'] == 'variance_5':
                kernel_initializers.append(initializers.VarianceScaling(scale=5, seed=np.random.seed()))
            elif config['kernel_initializer'] == 'variance_10':
                kernel_initializers.append(initializers.VarianceScaling(scale=10, seed=np.random.seed()))
            elif config['kernel_initializer'] == 'lecun_uniform':
                kernel_initializers.append(initializers.lecun_uniform(seed=np.random.seed()))
            elif config['kernel_initializer'] == 'lecun_normal':
                kernel_initializers.append(initializers.lecun_normal(seed=np.random.seed()))
            elif config['kernel_initializer'] == 'orthogonal':
                kernel_initializers.append(initializers.orthogonal(seed=np.random.seed()))

        activation = None
        if config['activation'] == 'linear':
            activation = None
        elif config['activation'] == 'relu':
            activation = tf.keras.activations.relu
        elif config['activation'] == 'relu6':
            activation = tf.keras.activations.relu(max_value=6)
        elif config['activation'] == 'sigmoid':
            activation = tf.keras.activations.sigmoid
        elif config['activation'] == 'softplus':
            activation = tf.keras.activations.softplus
        elif config['activation'] == 'soft_sign':
            activation = tf.keras.activations.softsign
        elif config['activation'] == 'silu':
            activation = tf.nn.silu
        elif config['activation'] == 'swish':
            activation = tf.nn.swish
        elif config['activation'] == 'log_sigmoid':
            activation = tf.math.log_sigmoid
        elif config['activation'] == 'hard_sigmoid':
            activation = tf.keras.activations.hard_sigmoid
        elif config['activation'] == 'hard_silu':
            activation = tf.keras.activations.hard_silu
        elif config['activation'] == 'hard_swish':
            activation = tf.keras.activations.hard_swish
        elif config['activation'] == 'hard_tanh':
            activation = tf.keras.activations.hard_tanh
        elif config['activation'] == 'elu':
            activation = tf.keras.activations.elu
        elif config['activation'] == 'celu':
            activation = tf.keras.activations.celu
        elif config['activation'] == 'selu':
            activation = tf.keras.activations.selu
        elif config['activation'] == 'gelu':
            activation = tf.nn.gelu
        elif config['activation'] == 'glu':
            activation = tf.keras.activations.glu

        self.inputs = tf.keras.layers.Input(shape=input_shape, name='my_input')
        self.has_conv_layers = len(config['conv_layers']) > 0
        self.has_dense_layers = config['dense_layers'] > 0
        if self.has_conv_layers:
            self.conv_layers = []
            for i, (filters, kernel_size, strides) in enumerate(config['conv_layers']):
                if config['conv_layers_noisy']:
                    # if i == 0:
                    #     self.conv_layers.append(NoisyConv2D(filters, kernel_size, strides=strides, kernel_initializer=kernel_initializers.pop(), activation=activation, input_shape=input_shape))
                    # else:
                    #     self.conv_layers.append(NoisyConv2D(filters, kernel_size, strides=strides, kernel_initializer=kernel_initializers.pop(), activation=activation))
                    pass
                else:
                    if i == 0:
                        self.conv_layers.append(tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, kernel_initializer=kernel_initializers.pop(), activation=activation, input_shape=input_shape, padding='same'))
                    else:
                        self.conv_layers.append(tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, kernel_initializer=kernel_initializers.pop(), activation=activation, padding='same'))
            self.conv_layers.append(tf.keras.layers.Flatten())

        if self.has_dense_layers:
            self.dense_layers = []
            for i in range(config['dense_layers']):
                if config['dense_layers_noisy']:
                    self.dense_layers.append(NoisyDense(config['width'], sigma=config['noisy_sigma'], kernel_initializer=kernel_initializers.pop(), activation=activation))
                else:
                    self.dense_layers.append(tf.keras.layers.Dense(config['width'], kernel_initializer=kernel_initializers.pop(), activation=activation))

        self.has_value_hidden_layers = config['value_hidden_layers'] > 0
        if self.has_value_hidden_layers:
            self.value_hidden_layers = []
            for i in range(config['value_hidden_layers']):
                self.value_hidden_layers.append(NoisyDense(config['width'], sigma=config['noisy_sigma'], kernel_initializer=kernel_initializers.pop(), activation=activation))

        self.value = NoisyDense(
            config["atom_size"], sigma=config['noisy_sigma'], kernel_initializer=kernel_initializers.pop(), activation="linear", name="HiddenV"
        )

        self.has_advantage_hidden_layers = config['advantage_hidden_layers'] > 0
        if self.has_advantage_hidden_layers:
            self.advantage_hidden_layers = []
            for i in range(config['advantage_hidden_layers']):
                self.advantage_hidden_layers.append(NoisyDense(config['width'], sigma=config['noisy_sigma'], kernel_initializer=kernel_initializers.pop(), activation=activation))

        self.advantage = NoisyDense(config["atom_size"] * output_size, sigma=config['noisy_sigma'], kernel_initializer=kernel_initializers.pop(), activation="linear", name="A")
        self.advantage_reduced_mean = tf.keras.layers.Lambda(
            lambda a: a - tf.reduce_mean(a, axis=1, keepdims=True), name="Ao"
        )

        self.advantage_reshaped = tf.keras.layers.Reshape((output_size, config["atom_size"]), name="ReshapeAo")
        self.value_reshaped = tf.keras.layers.Reshape((1, config["atom_size"]), name="ReshapeV")
        self.add = tf.keras.layers.Add()
        # self.softmax = tf.keras.activations.softmax(self.add, axis=-1)
        # ONLY CLIP FOR CATEGORICAL CROSS ENTROPY LOSS TO PREVENT NAN
        self.clip_qs = tf.keras.layers.Lambda(
            lambda q: tf.clip_by_value(q, 1e-3, 1), name="ClippedQ"
        )
        self.outputs = tf.keras.layers.Lambda(
            lambda q: tf.reduce_sum(q * config['support'], axis=2), name="Q"
        )

    def call(self, inputs, training=False):
        x = inputs
        if self.has_conv_layers:
            for layer in self.conv_layers:
                x = layer(x)
        if self.has_dense_layers:
            for layer in self.dense_layers:
                x = layer(x)
        if self.has_value_hidden_layers:
            for layer in self.value_hidden_layers:
                x = layer(x)
        value = self.value(x)
        value = self.value_reshaped(value)

        if self.has_advantage_hidden_layers:
            for layer in self.advantage_hidden_layers:
                x = layer(x)
        advantage = self.advantage(x)
        advantage = self.advantage_reduced_mean(advantage)
        advantage = self.advantage_reshaped(advantage)

        q = self.add([value, advantage])
        q = tf.keras.activations.softmax(q, axis=-1)
        # MIGHT BE ABLE TO REMOVE CLIPPING ENTIRELY SINCE I DONT THINK THE TENSORFLOW LOSSES CAN RETURN NaN
        # q = self.clip_qs(q)
        # q = self.outputs(q)
        return q

    def reset_noise(self):
        if self.has_dense_layers and self.config['conv_layers_noisy']:
            for layer in self.conv_layers:
                layer.reset_noise()
        if self.has_dense_layers and self.config['dense_layers_noisy']:
            for layer in self.dense_layers:
                layer.reset_noise()
        if self.has_value_hidden_layers:
            for layer in self.value_hidden_layers:
                layer.reset_noise()
        if self.has_advantage_hidden_layers:
            for layer in self.advantage_hidden_layers:
                layer.reset_noise()
        self.value.reset_noise()
        self.advantage.reset_noise()

class RainbowDQN:
    def __init__(
        self,
        env,
        model_name=datetime.datetime.now().timestamp(),
        config=None,
        start_episode=0,
    ):
        self.config = config
        self.model_name = model_name
        self.env = env
        self.test_env = copy.deepcopy(env)
        self.observation_dimensions = env.observation_space.shape
        self.num_actions = env.action_space.n

        self.model = Network(config, self.num_actions, input_shape=self.observation_dimensions)

        self.target_model = Network(config, self.num_actions, input_shape=self.observation_dimensions)

        self.optimizer = config["optimizer_function"]
        self.adam_epsilon=config["adam_epsilon"]
        self.learning_rate = config["learning_rate"]
        self.loss_function = config["loss_function"]
        self.clipnorm = 10.0

        self.model.compile(
            optimizer=self.optimizer(learning_rate=self.learning_rate, epsilon=self.adam_epsilon, clipnorm=self.clipnorm),
            loss=config["loss_function"],
        )

        self.target_model.compile(
            optimizer=self.optimizer(learning_rate=self.learning_rate, epsilon=self.adam_epsilon, clipnorm=self.clipnorm),
            loss=config["loss_function"],
        )

        self.target_model.set_weights(self.model.get_weights())

        self.num_training_steps = int(config["num_training_steps"])
        self.start_episode = start_episode

        self.discount_factor = config["discount_factor"]

        self.replay_batch_size = int(config["replay_batch_size"])
        self.replay_period = int(config["replay_period"])
        self.memory_size = max(int(config["memory_size"]), self.replay_batch_size)
        self.min_memory_size = int(config["min_memory_size"])

        self.soft_update = config["soft_update"]
        self.transfer_frequency = int(config["transfer_frequency"])
        self.ema_beta = config["ema_beta"]

        self.per_beta = config["per_beta"]
        # self.per_beta_increase = config["per_beta_increase"]
        self.per_beta_increase = (1 - self.per_beta) / self.num_training_steps
        self.per_epsilon = config["per_epsilon"]
        # TESTING WITH FAST PRIORITIZED EXPERIENCE REPLAY
        # it is an approximation but should be much faster computationally
        self.memory = PrioritizedReplayBuffer(
            observation_dimensions=self.observation_dimensions,
            max_size=self.memory_size,
            batch_size=self.replay_batch_size,
            max_priority=1.0,
            alpha=config["per_alpha"],
            # epsilon=config["per_epsilon"],
            n_step=config["n_step"],
            gamma=config["discount_factor"],
        )

        self.use_n_step = config["n_step"] > 1

        self.n_step = config["n_step"]

        if self.use_n_step:
            self.memory_n = ReplayBuffer(
                observation_dimensions=self.observation_dimensions,
                max_size=self.memory_size,
                batch_size=self.replay_batch_size,
                n_step=self.n_step,
                gamma=config["discount_factor"],
            )

        self.v_min = config["v_min"]
        self.v_max = config["v_max"]

        self.atom_size = config["atom_size"]
        self.support = np.linspace(self.v_min, self.v_max, self.atom_size)

        self.transition = list()
        self.is_test = True
        # self.search = search.Search(
        #     scoring_function=self.score_state,
        #     max_depth=config["search_max_depth"],
        #     max_time=config["search_max_time"],
        #     transposition_table=search.TranspositionTable(
        #         buckets=config["search_transposition_table_buckets"],
        #         bucket_size=config["search_transposition_table_bucket_size"],
        #         replacement_strategy=search.TranspositionTable.replacement_strategies[
        #             config["search_transposition_table_replacement_strategy"]
        #         ],
        #     ),
        #     debug=False,
        # )

    def export(self, episode=-1, best_model=False):
        if episode != -1:
            path = "./{}_{}_episodes.keras".format(
                self.model_name, episode + self.start_episode
            )
        else:
            path = "./{}.keras".format(self.model_name)

        if best_model:
            path = "./best_model.keras"

        self.model.save(path)

    def prepare_states(self, state):
        if (self.env.observation_space.high == 255).all():
            state = np.array(state)/255
        # print(state.shape)
        if state.shape == self.observation_dimensions:
            new_shape = (1,) + state.shape
            state_input = state.reshape(new_shape)
        else:
            state_input = state
        # print(state_input.shape)
        # observation_high = self.env.observation_space.high
        # observation_low = self.env.observation_space.low
        # for s in state_input:
        #     for i in range(len(s)):
        #         s[i] = s[i] - observation_low[i]
        #         s[i] = s[i] / (observation_high[i] - observation_low[i])
        # print(state_input)
        # NORMALIZE VALUES
        return state_input

    def predict_single(self, state):
        state_input = self.prepare_states(state)
        # print(state_input)
        q_values = self.model(inputs=state_input).numpy()
        return q_values

    def select_action(self, state):
        q_values = np.sum(np.multiply(self.predict_single(state), np.array(self.support)), axis=2)
        # print(q_values)
        selected_action = np.argmax(q_values)
        # selected_action = np.argmax(self.predict_single(state))
        if not self.is_test:
            self.transition = [state, selected_action]
        return selected_action

    def step(self, action):
        if not self.is_test:
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.transition += [reward, next_state, done]
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            else:
                one_step_transition = self.transition

            if one_step_transition:
                self.memory.store(*one_step_transition)
        else:
            next_state, reward, terminated, truncated, _ = self.test_env.step(action)

        return next_state, reward, terminated, truncated

    def experience_replay(self):
        # print("Experience Replay")
        # time1 = 0
        # time1 = time()
        with tf.GradientTape() as tape:
            # print("One Step Learning")
            # time2 = 0
            # time2 = time()
            elementwise_loss = 0
            samples = self.memory.sample(self.per_beta)
            actions = samples["actions"]
            observations = samples["observations"]
            inputs = self.prepare_states(observations)
            weights = samples["weights"].reshape(-1, 1)
            # print("weights", weights)
            indices = samples["indices"]
            discount_factor = self.discount_factor
            target_ditributions = self.compute_target_distributions(samples, discount_factor)
            self.model.loss.actions = samples["actions"]
            initial_distributions = self.model(inputs)
            distributions_to_train = tf.gather_nd(initial_distributions, list(zip(range(initial_distributions.shape[0]), actions)))
            elementwise_loss = self.model.loss.call(y_pred=distributions_to_train, y_true=tf.convert_to_tensor
            (target_ditributions))
            assert np.all(elementwise_loss) >= 0, "Elementwise Loss: {}".format(elementwise_loss)
            # print("One Step Learning Time ", time() - time2)
            if self.use_n_step:
                # print("N-Step Learning")
                # time2 = time()
                discount_factor = self.discount_factor ** self.n_step
                n_step_samples = self.memory_n.sample_from_indices(indices)
                actions = n_step_samples["actions"]
                n_step_observations = n_step_samples["observations"]
                observations = n_step_observations
                inputs = self.prepare_states(observations)
                target_ditributions = self.compute_target_distributions(n_step_samples, discount_factor)
                self.model.loss.actions = n_step_samples["actions"]
                initial_distributions = self.model(inputs)
                distributions_to_train = tf.gather_nd(initial_distributions, list(zip(range(initial_distributions.shape[0]), actions)))
                elementwise_loss_n_step = self.model.loss.call(y_pred=distributions_to_train, y_true=tf.convert_to_tensor(target_ditributions))
                # add the losses together to reduce variance (original paper just uses n_step loss)
                elementwise_loss += elementwise_loss_n_step
                assert np.all(elementwise_loss) >= 0, "Elementwise Loss: {}".format(elementwise_loss)
                # print("Elementwise Loss N-Step Shape", elementwise_loss_n_step.shape)
                # print("N-Step Learning Time ", time() - time2)

            # print(weights)
            loss = tf.reduce_mean(elementwise_loss * weights)

        #TRAINING WITH GRADIENT TAPE
        # print("Computing Gradients")
        # time2 = time()
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # print("Computing Gradients Time ", time() - time2)
        # print("Applying Gradients")
        # time2 = time()
        self.optimizer(learning_rate=self.learning_rate, epsilon=self.adam_epsilon, clipnorm=self.clipnorm).apply_gradients(grads_and_vars=zip(gradients, self.model.trainable_variables))
        # print("Applying Gradients Time ", time() - time2)

        # TRAINING WITH tf.train_on_batch
        # print("Training Model on Batch")
        # loss = self.model.train_on_batch(samples["observations"], target_ditributions, sample_weight=weights)

        # print("Updating Priorities")
        # time2 = time()
        prioritized_loss = elementwise_loss + self.per_epsilon
        # CLIPPING PRIORITIZED LOSS FOR ROUNDING ERRORS OR NEGATIVE LOSSES (IDK HOW WE ARE GETTING NEGATIVE LSOSES)
        prioritized_loss = np.clip(prioritized_loss, 0.01, prioritized_loss.max())
        self.memory.update_priorities(indices, prioritized_loss)
        # print("Updating Priorities Time ", time() - time2)

        # print("Resetting Noise")
        # time2 = time()
        self.model.reset_noise()
        self.target_model.reset_noise()
        # print("Resetting Noise Time ", time() - time2)

        loss = loss.numpy()
        # print("Experience Replay Time ", time() - time1)
        return loss

    def compute_target_distributions(self, samples, discount_factor):
        # print("Computing Target Distributions")
        # time1 = 0
        # time1 = time()
        observations = samples["observations"]
        inputs = self.prepare_states(observations)
        next_observations = samples["next_observations"]
        next_inputs = self.prepare_states(next_observations)
        rewards = samples["rewards"].reshape(-1,1)
        dones = samples["dones"].reshape(-1,1)

        # print(rewards.shape, dones.shape)

        next_actions = np.argmax(np.sum(self.model(inputs).numpy(), axis=2), axis=1)
        target_network_distributions = self.target_model(next_inputs).numpy()

        target_distributions = target_network_distributions[range(self.replay_batch_size), next_actions]
        target_z = rewards + (1 - dones) * (discount_factor) * self.support
        # print("Target Z", target_z.shape)
        target_z = np.clip(target_z, self.v_min, self.v_max)

        b = ((target_z - self.v_min) / (self.v_max - self.v_min)) * (self.atom_size - 1)
        # print(b)
        l, u = tf.cast(tf.math.floor(b), tf.int32), tf.cast(tf.math.ceil(b), tf.int32)
        # print(l, u)
        m = np.zeros_like(target_distributions)
        assert m.shape == l.shape
        lower_distributions = target_distributions * (tf.cast(u, tf.float64) - b)
        upper_distributions = target_distributions * (b - tf.cast(l, tf.float64))

        for i in range(self.replay_batch_size):
            np.add.at(m[i], np.asarray(l)[i], lower_distributions[i])
            np.add.at(m[i], np.asarray(u)[i], upper_distributions[i])
            # print(m[i])
        # target_distributions = np.clip(m, 1e-3, 1)
        target_distributions = m
        # print("Computing Target Distributions Time ", time() - time1)
        return target_distributions

    # def score_state(self, state, turn):
    #     state_input = self.prepare_state(state)
    #     q = self.predict(state_input)

    #     if (turn % 2) == 0:
    #         return q.max(), q.argmax()

    #     return q.min(), q.argmin()

    # def play_optimal_move(
    #     self, state: bb.Bitboard, turn: int, max_depth: int, with_output=True
    # ):
    #     # q_value, action = self.alpha_beta_pruning(state, turn, max_depth=max_depth)
    #     q_value, action = self.search.iterative_deepening(state, turn, max_depth)
    #     if with_output:
    #         print("Evaluation: {}".format(q_value))
    #         print("Action: {}".format(action + 1))
    #     state.move(turn % 2, action)
    #     winner, _ = state.check_victory()

    #     if winner == 0:
    #         return False
    #     else:
    #         return True

    def action_mask(self, q, state, turn):
        q_copy = copy.deepcopy(q)
        for i in range(len(q_copy)):
            if not state.is_valid_move(i):
                if turn % 2 == 0:
                    q_copy[i] = float("-inf")
                else:
                    q_copy[i] = float("inf")
        return q_copy

    def fill_memory(self):
        state, _ = self.env.reset()
        # print(state)
        for experience in range(self.min_memory_size):
            # clear_output(wait=False)
            # print("Filling Memory")
            print("Memory Size: {}/{}".format(experience, self.min_memory_size))
            # state_input = self.prepare_state(state)
            action = self.env.action_space.sample()
            self.transition = [state, action]

            next_state, reward, terminated, truncated = self.step(action)
            done = terminated or truncated
            state = next_state
            if done:
                state, _ = self.env.reset()

    def update_target_model(self, step):
        # print("Updating Target Model")
        # time1 = 0
        # time1 = time()
        if self.soft_update:
            new_weights = self.target_model.get_weights()

            counter = 0
            for wt, wp in zip(
                self.target_model.get_weights(),
                self.model.get_weights(),
            ):
                wt = (self.ema_beta * wt) + ((1 - self.ema_beta) * wp)
                new_weights[counter] = wt
                counter += 1
            self.target_model.set_weights(new_weights)
        else:
            if step % self.transfer_frequency == 0 and (len(self.memory) >= self.replay_batch_size):
                self.target_model.set_weights(self.model.get_weights())
        # print("Updating Target Model Time ", time() - time1)

    def train(self, graph_interval=200):
        self.is_test = False
        stat_score = [] # make these num trials divided by graph interval so i dont need to append (to make it faster?)
        stat_test_score = []
        stat_loss = []
        self.fill_memory()
        num_trials_truncated = 0
        state, _ = self.env.reset()
        model_update_count = 0
        score = 0
        for step in range(self.num_training_steps):
            # state_input = self.prepare_state(state)
            # clear_output(wait=False)
            print("{} Step: {}/{}".format(self.model_name, step, self.num_training_steps))
            # print("Last Training Score: ", stat_score[-1] if len(stat_score) > 0 else 0)
            # print("Last Training Loss: ", stat_loss[-1] if len(stat_loss) > 0 else 0)
            action = self.select_action(state)

            next_state, reward, terminated, truncated = self.step(action)
            done = terminated or truncated
            state = next_state
            score += reward

            if truncated:
                num_trials_truncated += 1
            #     if num_trials_truncated > 100:
            #         num_trials_truncated += self.num_training_steps - step
            #         break
            self.per_beta = min(1.0, self.per_beta + self.per_beta_increase)

            if done:
                state, _ = self.env.reset()
                stat_score.append(score)
                if score >= self.env.spec.reward_threshold:
                    print("Your DQN agent has achieved the env's reward threshold.")
                    # test_score = self.test()
                    # if test_score >= self.env.spec.reward_threshold:
                    #     print("Congratulations!")
                    #     break
                    # else:
                    #     print("It was a fluke!")
                score = 0

            if (step % self.replay_period) == 0 and (len(self.memory) >= self.replay_batch_size):
                model_update_count += 1
                loss = self.experience_replay()
                stat_loss.append(loss)

                self.update_target_model(model_update_count)


            if step % graph_interval == 0 and step > 0:
                self.export()
                # stat_test_score.append(self.test())
                self.plot_graph(stat_score, stat_loss, stat_test_score, step)

        self.plot_graph(stat_score, stat_loss, stat_test_score, step)
        self.export()
        self.env.close()
        return num_trials_truncated / self.num_training_steps

    def plot_graph(self, score, loss, test_score, step):
            fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(30, 5))
            ax1.plot(score, linestyle="solid")
            ax1.set_title('Frame {}. Score: {}'.format(step, np.mean(score[-10:])))
            ax2.plot(loss, linestyle="solid")
            ax2.set_title('Frame {}. Loss: {}'.format(step, np.mean(loss[-10:])))
            ax3.plot(test_score, linestyle="solid")
            ax3.axhline(y=self.env.spec.reward_threshold, color='r', linestyle='-')
            ax3.set_title('Frame {}. Test Score: {}'.format(step, np.mean(test_score[-10:])))
            plt.savefig("./{}.png".format(self.model_name))
            plt.close(fig)

    def test(self, video_folder = '', num_trials=100) -> None:
        """Test the agent."""
        self.is_test = True
        average_score = 0
        for trials in range(num_trials - 1):
            state, _ = self.test_env.reset()
            done = False
            score = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated = self.step(action)
                done = terminated or truncated
                state = next_state

                score += reward
            average_score += score
            print("score: ", score)

        if video_folder == '':
            video_folder = "./videos/{}".format(self.model_name)
        # for recording a video
        self.test_env = gym.wrappers.RecordVideo(self.test_env, video_folder)
        state, _ = self.test_env.reset()
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, terminated, truncated = self.step(action)
            done = terminated or truncated
            state = next_state

            score += reward

        print("score: ", score)
        average_score += score
        self.test_env.close()

        # reset
        self.is_test = False
        average_score /= num_trials
        return average_score