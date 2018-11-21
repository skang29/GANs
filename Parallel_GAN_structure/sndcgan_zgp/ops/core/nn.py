""""
core/nn under tensorflow environment.
Supports NCCL multi-gpu environment.
To activate the environment, use code below in your
main.py.
>> os.environ['nccl_multigpu_env'] = 'true'
"""
__version__ = "1.0.0"

import os
import tensorflow as tf
from tensorflow.contrib.nccl.ops import gen_nccl_ops
NCCL_FLAG = os.environ.get('nccl_multigpu_env')

__all__ = ['moments', 'weighted_moments']


if NCCL_FLAG == 'true':
    # NCCL multi-gpu environment.
    from tensorflow.python.ops import array_ops
    from tensorflow.python.ops import math_ops
    from tensorflow.python.framework import dtypes
    from tensorflow.python.framework import ops


    def moments(
            x,
            axes,
            tower_config,
            shift=None,  # pylint: disable=unused-argument
            name=None,
            keep_dims=False):
        """Calculate the mean and variance of `x`.
        The mean and variance are calculated by aggregating the contents of `x`
        across `axes`.  If `x` is 1-D and `axes = [0]` this is just the mean
        and variance of a vector.
        Note: shift is currently not used; the true mean is computed and used.
        When using these moments for batch normalization (see
        `tf.nn.batch_normalization`):
         * for so-called "global normalization", used with convolutional filters with
           shape `[batch, height, width, depth]`, pass `axes=[0, 1, 2]`.
         * for simple batch normalization pass `axes=[0]` (batch only).
        Args:
          x: A `Tensor`.
          axes: Array of ints.  Axes along which to compute mean and
            variance.
          shift: Not used in the current implementation
          name: Name used to scope the operations that compute the moments.
          keep_dims: produce moments with the same dimensionality as the input.
        Returns:
          Two `Tensor` objects: `mean` and `variance`.
        """
        with ops.name_scope(name, "moments", [x, axes]):
            nccl_name = "NCCL" if not tower_config.is_test else "NCCL_TEST"

            # The dynamic range of fp16 is too limited to support the collection of
            # sufficient statistics. As a workaround we simply perform the operations
            # on 32-bit floats before converting the mean and variance back to fp16
            y = math_ops.cast(x, dtypes.float32) if x.dtype == dtypes.float16 else x
            # Compute true mean while keeping the dims for proper broadcasting.

            # Original Code: mean = math_ops.reduce_mean(y, axes, keepdims=True, name="mean")

            device_mean = math_ops.reduce_mean(y, axes, keepdims=True, name="mean")

            shared_name = device_mean.name. \
                replace(tower_config.name, tower_config.prefix.format(nccl_name))

            mean = gen_nccl_ops.nccl_all_reduce(
                input=device_mean,
                reduction="sum",
                num_devices=tower_config.num_devices,
                shared_name=shared_name
            ) / (1.0 * tower_config.num_devices)

            # sample variance, not unbiased variance
            # Note: stop_gradient does not change the gradient that gets
            #       backpropagated to the mean from the variance calculation,
            #       because that gradient is zero

            # Original Code: variance = math_ops.reduce_mean(
            #     math_ops.squared_difference(y, array_ops.stop_gradient(mean)),
            #     axes,
            #     keepdims=True,
            #     name="variance")

            device_variance = math_ops.reduce_mean(
                math_ops.squared_difference(y, array_ops.stop_gradient(mean)),
                axes,
                keepdims=True,
                name="variance"
            )

            shared_name = device_variance.name. \
                replace(tower_config.name, tower_config.prefix.format(nccl_name))

            variance = gen_nccl_ops.nccl_all_reduce(
                input=device_variance,
                reduction="sum",
                num_devices=tower_config.num_devices,
                shared_name=shared_name
            ) / (1.0 * tower_config.num_devices)

            if not keep_dims:
                mean = array_ops.squeeze(mean, axes)
                variance = array_ops.squeeze(variance, axes)
            if x.dtype == dtypes.float16:
                return (math_ops.cast(mean, dtypes.float16),
                        math_ops.cast(variance, dtypes.float16))
            else:
                return (mean, variance)


    def weighted_moments(x, axes, frequency_weights, tower_config, name=None, keep_dims=False):
        """Returns the frequency-weighted mean and variance of `x`.
        Args:
          x: A tensor.
          axes: 1-d tensor of int32 values; these are the axes along which
            to compute mean and variance.
          frequency_weights: A tensor of positive weights which can be
            broadcast with x.
          name: Name used to scope the operation.
          keep_dims: Produce moments with the same dimensionality as the input.
        Returns:
          Two tensors: `weighted_mean` and `weighted_variance`.
        """
        with ops.name_scope(name, "weighted_moments", [x, frequency_weights, axes]):
            x = ops.convert_to_tensor(x, name="x")
            frequency_weights = ops.convert_to_tensor(
                frequency_weights, name="frequency_weights")

            # Unlike moments(), this just uses a simpler two-pass method.

            # See comment in moments() WRT precision; it applies here too.
            needs_cast = x.dtype == dtypes.float16
            if needs_cast:
                x = math_ops.cast(x, dtypes.float32)

            if frequency_weights.dtype != x.dtype:
                frequency_weights = math_ops.cast(frequency_weights, x.dtype)

            # Note that we use keep_dims=True for our reductions regardless of the arg;
            # this is so that the results remain broadcast-compatible with the inputs.

            # Original Code: weighted_input_sum = math_ops.reduce_sum(
            #     frequency_weights * x, axes, name="weighted_input_sum", keepdims=True)

            nccl_name = "NCCL" if not tower_config.is_test else "NCCL_TEST"
            shared_name = tf.get_variable_scope().name. \
                replace(tower_config.name, tower_config.prefix.format(nccl_name))
            device_weighted_input_sum = math_ops.reduce_sum(
                    frequency_weights * x, axes, name="weighted_input_sum", keepdims=True)
            weighted_input_sum = gen_nccl_ops.nccl_all_reduce(
                input=device_weighted_input_sum,
                reduction="sum",
                num_devices=tower_config.num_devices,
                shared_name=shared_name
            ) / (1.0 * tower_config.num_devices)

            # The shape of the weights isn't necessarily the same as x's
            # shape, just broadcast-compatible with it -- so this expression
            # performs broadcasting to give a per-item weight, with the same
            # shape as (freqency_weights * x). This avoids having to reason
            # through all the broadcast logic to compute a correct
            # sum_of_weights.
            broadcasted_weights = frequency_weights + array_ops.zeros_like(x)

            sum_of_weights = math_ops.reduce_sum(
                broadcasted_weights, axes, name="sum_of_weights", keepdims=True)

            divisor = math_ops.reciprocal(sum_of_weights, name="inv_weight_sum")

            weighted_mean = math_ops.multiply(weighted_input_sum, divisor)

            # Have the weighted mean; now on to variance:
            # weighted_distsq = math_ops.reduce_sum(
            #     frequency_weights * math_ops.squared_difference(x, weighted_mean),
            #     axes,
            #     name="weighted_distsq",
            #     keepdims=True)

            nccl_name = "NCCL" if not tower_config.is_test else "NCCL_TEST"
            shared_name = tf.get_variable_scope().name. \
                replace(tower_config.name, tower_config.prefix.format(nccl_name))
            device_weighted_distsq = math_ops.reduce_sum(
                frequency_weights * math_ops.squared_difference(x, weighted_mean),
                axes,
                name="weighted_distsq",
                keepdims=True)
            weighted_distsq = gen_nccl_ops.nccl_all_reduce(
                input=device_weighted_distsq,
                reduction="sum",
                num_devices=tower_config.num_devices,
                shared_name=shared_name
            ) / (1.0 * tower_config.num_devices)

            weighted_variance = math_ops.multiply(weighted_distsq, divisor)

            if not keep_dims:
                weighted_mean = array_ops.squeeze(weighted_mean, axis=axes)
                weighted_variance = array_ops.squeeze(
                    weighted_variance, axis=axes)

            if needs_cast:
                weighted_mean = math_ops.cast(weighted_mean, dtypes.float16)
                weighted_variance = math_ops.cast(weighted_variance, dtypes.float16)

            return weighted_mean, weighted_variance

else:
    # None NCCL multi-gpu environment.
    def moments(
            x,
            axes,
            tower_config,
            shift=None,  # pylint: disable=unused-argument
            name=None,
            keep_dims=False):
        return tf.nn.moments(x=x, axes=axes, shift=shift, name=name, keep_dims=keep_dims)

    def weighted_moments(x, axes, frequency_weights, tower_config, name=None, keep_dims=False):
        return tf.nn.weighted_moments(x=x,
                                      axes=axes,
                                      frequency_weights=frequency_weights,
                                      name=name,
                                      keep_dims=keep_dims)
