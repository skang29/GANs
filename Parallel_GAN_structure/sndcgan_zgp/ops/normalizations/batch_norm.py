""""
Normalizations / batch_norm under tensorflow environment.
Supports NCCL multi-gpu environment.
To activate the environment, use code below in your
main.py.
>> os.environ['nccl_multigpu_env'] = 'true'
"""
__version__ = "1.0.0"

import os
import tensorflow as tf

NCCL_FLAG = os.environ.get('nccl_multigpu_env')

__all__ = ['batch_norm']


if NCCL_FLAG == 'true':
    # NCCL multi-gpu environment.
    def batch_norm(x, tower_config, is_training=True, name="batch_norm"):
        return batch_norm_backbone(x,
                                   decay=0.9,
                                   updates_collections=None,
                                   epsilon=1e-5,
                                   scale=True,
                                   is_training=is_training,
                                   scope=name,
                                   tower_config=tower_config)


    from tensorflow.contrib.framework.python.ops import add_arg_scope
    from tensorflow.contrib.framework.python.ops import variables
    from tensorflow.contrib.layers.python.layers import utils
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import array_ops
    from tensorflow.python.ops import init_ops
    from tensorflow.python.ops import nn
    from tensorflow.python.ops import variable_scope
    from tensorflow.python.training import moving_averages
    from tensorflow.contrib.layers.python.layers.layers import _build_variable_getter

    from sndcgan_zgp.ops.core.nn import moments, weighted_moments

    DATA_FORMAT_NCHW = 'NCHW'
    DATA_FORMAT_NHWC = 'NHWC'
    DATA_FORMAT_NCDHW = 'NCDHW'
    DATA_FORMAT_NDHWC = 'NDHWC'

    @add_arg_scope
    def batch_norm_backbone(inputs,
                            decay=0.999,
                            center=True,
                            scale=False,
                            epsilon=0.001,
                            activation_fn=None,
                            param_initializers=None,
                            param_regularizers=None,
                            updates_collections=ops.GraphKeys.UPDATE_OPS,
                            is_training=True,
                            reuse=None,
                            variables_collections=None,
                            outputs_collections=None,
                            trainable=True,
                            batch_weights=None,
                            fused=None,
                            data_format=DATA_FORMAT_NHWC,
                            zero_debias_moving_mean=False,
                            scope=None,
                            renorm=False,
                            renorm_clipping=None,
                            renorm_decay=0.99,
                            adjustment=None,
                            tower_config=None):

        """Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167.
          "Batch Normalization: Accelerating Deep Network Training by Reducing
          Internal Covariate Shift"
          Sergey Ioffe, Christian Szegedy
        Can be used as a normalizer function for conv2d and fully_connected. The
        normalization is over all but the last dimension if `data_format` is `NHWC`
        and all but the second dimension if `data_format` is `NCHW`.  In case of a 2D
        tensor this corresponds to the batch dimension, while in case of a 4D tensor
        this
        corresponds to the batch and space dimensions.
        Note: when training, the moving_mean and moving_variance need to be updated.
        By default the update ops are placed in `tf.GraphKeys.UPDATE_OPS`, so they
        need to be added as a dependency to the `train_op`. For example:
        ```python
          update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
          with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)
        ```
        One can set updates_collections=None to force the updates in place, but that
        can have a speed penalty, especially in distributed settings.
        Args:
          inputs: A tensor with 2 or more dimensions, where the first dimension has
            `batch_size`. The normalization is over all but the last dimension if
            `data_format` is `NHWC` and the second dimension if `data_format` is
            `NCHW`.
          decay: Decay for the moving average. Reasonable values for `decay` are close
            to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
            Lower `decay` value (recommend trying `decay`=0.9) if model experiences
            reasonably good training performance but poor validation and/or test
            performance. Try zero_debias_moving_mean=True for improved stability.
          center: If True, add offset of `beta` to normalized tensor. If False, `beta`
            is ignored.
          scale: If True, multiply by `gamma`. If False, `gamma` is
            not used. When the next layer is linear (also e.g. `nn.relu`), this can be
            disabled since the scaling can be done by the next layer.
          epsilon: Small float added to variance to avoid dividing by zero.
          activation_fn: Activation function, default set to None to skip it and
            maintain a linear activation.
          param_initializers: Optional initializers for beta, gamma, moving mean and
            moving variance.
          param_regularizers: Optional regularizer for beta and gamma.
          updates_collections: Collections to collect the update ops for computation.
            The updates_ops need to be executed with the train_op.
            If None, a control dependency would be added to make sure the updates are
            computed in place.
          is_training: Whether or not the layer is in training mode. In training mode
            it would accumulate the statistics of the moments into `moving_mean` and
            `moving_variance` using an exponential moving average with the given
            `decay`. When it is not in training mode then it would use the values of
            the `moving_mean` and the `moving_variance`.
          reuse: Whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
          variables_collections: Optional collections for the variables.
          outputs_collections: Collections to add the outputs.
          trainable: If `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
          batch_weights: An optional tensor of shape `[batch_size]`,
            containing a frequency weight for each batch item. If present,
            then the batch normalization uses weighted mean and
            variance. (This can be used to correct for bias in training
            example selection.)
          fused: if `None` or `True`, use a faster, fused implementation if possible.
            If `False`, use the system recommended implementation.
          data_format: A string. `NHWC` (default) and `NCHW` are supported.
          zero_debias_moving_mean: Use zero_debias for moving_mean. It creates a new
            pair of variables 'moving_mean/biased' and 'moving_mean/local_step'.
          scope: Optional scope for `variable_scope`.
          renorm: Whether to use Batch Renormalization
            (https://arxiv.org/abs/1702.03275). This adds extra variables during
            training. The inference is the same for either value of this parameter.
          renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
            scalar `Tensors` used to clip the renorm correction. The correction
            `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
            `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
            dmax are set to inf, 0, inf, respectively.
          renorm_decay: Momentum used to update the moving means and standard
            deviations with renorm. Unlike `momentum`, this affects training
            and should be neither too small (which would add noise) nor too large
            (which would give stale estimates). Note that `decay` is still applied
            to get the means and variances for inference.
          adjustment: A function taking the `Tensor` containing the (dynamic) shape of
            the input tensor and returning a pair (scale, bias) to apply to the
            normalized values (before gamma and beta), only during training. For
            example,
              `adjustment = lambda shape: (
                tf.random_uniform(shape[-1:], 0.93, 1.07),
                tf.random_uniform(shape[-1:], -0.1, 0.1))`
            will scale the normalized value by up to 7% up or down, then shift the
            result by up to 0.1 (with independent scaling and bias for each feature
            but shared across all examples), and finally apply gamma and/or beta. If
            `None`, no adjustment is applied.
        Returns:
          A `Tensor` representing the output of the operation.
        Raises:
          ValueError: If `data_format` is neither `NHWC` nor `NCHW`.
          ValueError: If the rank of `inputs` is undefined.
          ValueError: If rank or channels dimension of `inputs` is undefined.
        """
        # if fused is None:
        #     fused = True

        # Only use _fused_batch_norm if all of the following three
        # conditions are true:
        # (1) fused is set True;
        # (2) it is possible to use (currently it doesn't support batch weights,
        #   renorm, and the case when rank is neither 2 nor 4);
        # (3) it is used with zero_debias_moving_mean, or an input shape of rank 2,
        #   or non-default updates_collections (not implemented in
        #   normalization_layers.BatchNormalization yet); otherwise use the fused
        #   implementation in normalization_layers.BatchNormalization.
        # inputs = ops.convert_to_tensor(inputs)
        # rank = inputs.get_shape().ndims
        # possible_to_fuse = (
        #     batch_weights is None and not renorm and rank in [2, 4] and
        #     adjustment is None)
        # if fused and possible_to_fuse and (
        #                 zero_debias_moving_mean or rank == 2 or
        #                 updates_collections is not ops.GraphKeys.UPDATE_OPS):
        #     return _fused_batch_norm(
        #         inputs,
        #         decay=decay,
        #         center=center,
        #         scale=scale,
        #         epsilon=epsilon,
        #         activation_fn=activation_fn,
        #         param_initializers=param_initializers,
        #         param_regularizers=param_regularizers,
        #         updates_collections=updates_collections,
        #         is_training=is_training,
        #         reuse=reuse,
        #         variables_collections=variables_collections,
        #         outputs_collections=outputs_collections,
        #         trainable=trainable,
        #         data_format=data_format,
        #         zero_debias_moving_mean=zero_debias_moving_mean,
        #         scope=scope)

        if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
            raise ValueError('data_format has to be either NCHW or NHWC.')

        layer_variable_getter = _build_variable_getter()
        with variable_scope.variable_scope(
                scope,
                'BatchNorm', [inputs],
                reuse=reuse,
                custom_getter=layer_variable_getter) as sc:
            inputs = ops.convert_to_tensor(inputs)

            # # Determine whether we can use the core layer class.
            # if (batch_weights is None and
            #             updates_collections is ops.GraphKeys.UPDATE_OPS and
            #         not zero_debias_moving_mean):
            #     print("FUCK !!!!")
            #     # Use the core layer class.
            #     axis = 1 if data_format == DATA_FORMAT_NCHW else -1
            #     if not param_initializers:
            #         param_initializers = {}
            #     beta_initializer = param_initializers.get('beta',
            #                                               init_ops.zeros_initializer())
            #     gamma_initializer = param_initializers.get('gamma',
            #                                                init_ops.ones_initializer())
            #     moving_mean_initializer = param_initializers.get(
            #         'moving_mean', init_ops.zeros_initializer())
            #     moving_variance_initializer = param_initializers.get(
            #         'moving_variance', init_ops.ones_initializer())
            #     if not param_regularizers:
            #         param_regularizers = {}
            #     beta_regularizer = param_regularizers.get('beta')
            #     gamma_regularizer = param_regularizers.get('gamma')
            #     layer = normalization_layers.BatchNormalization(
            #         axis=axis,
            #         momentum=decay,
            #         epsilon=epsilon,
            #         center=center,
            #         scale=scale,
            #         beta_initializer=beta_initializer,
            #         gamma_initializer=gamma_initializer,
            #         moving_mean_initializer=moving_mean_initializer,
            #         moving_variance_initializer=moving_variance_initializer,
            #         beta_regularizer=beta_regularizer,
            #         gamma_regularizer=gamma_regularizer,
            #         trainable=trainable,
            #         renorm=renorm,
            #         renorm_clipping=renorm_clipping,
            #         renorm_momentum=renorm_decay,
            #         adjustment=adjustment,
            #         name=sc.name,
            #         _scope=sc,
            #         _reuse=reuse,
            #         fused=fused)
            #     outputs = layer.apply(inputs, training=is_training)
            #
            #     # Add variables to collections.
            #     _add_variable_to_collections(layer.moving_mean, variables_collections,
            #                                  'moving_mean')
            #     _add_variable_to_collections(layer.moving_variance, variables_collections,
            #                                  'moving_variance')
            #     if layer.beta is not None:
            #         _add_variable_to_collections(layer.beta, variables_collections, 'beta')
            #     if layer.gamma is not None:
            #         _add_variable_to_collections(layer.gamma, variables_collections,
            #                                      'gamma')
            #
            #     if activation_fn is not None:
            #         outputs = activation_fn(outputs)
            #     return utils.collect_named_outputs(outputs_collections, sc.name, outputs)

            # Not supported by layer class: batch_weights argument,
            # and custom updates_collections. In that case, use the legacy BN
            # implementation.
            # Custom updates collections are not supported because the update logic
            # is different in this case, in particular w.r.t. "forced updates" and
            # update op reuse.
            if renorm:
                raise ValueError('renorm is not supported with batch_weights, '
                                 'updates_collections or zero_debias_moving_mean')
            inputs_shape = inputs.get_shape()
            inputs_rank = inputs_shape.ndims
            if inputs_rank is None:
                raise ValueError('Inputs %s has undefined rank.' % inputs.name)
            dtype = inputs.dtype.base_dtype
            if batch_weights is not None:
                batch_weights = ops.convert_to_tensor(batch_weights)
                inputs_shape[0:1].assert_is_compatible_with(batch_weights.get_shape())
                # Reshape batch weight values so they broadcast across inputs.
                nshape = [-1] + [1 for _ in range(inputs_rank - 1)]
                batch_weights = array_ops.reshape(batch_weights, nshape)

            if data_format == DATA_FORMAT_NCHW:
                moments_axes = [0] + list(range(2, inputs_rank))
                params_shape = inputs_shape[1:2]
                # For NCHW format, rather than relying on implicit broadcasting, we
                # explicitly reshape the params to params_shape_broadcast when computing
                # the moments and the batch normalization.
                params_shape_broadcast = list(
                    [1, inputs_shape[1].value] + [1 for _ in range(2, inputs_rank)])
            else:
                moments_axes = list(range(inputs_rank - 1))
                params_shape = inputs_shape[-1:]
                params_shape_broadcast = None
            if not params_shape.is_fully_defined():
                raise ValueError('Inputs %s has undefined channels dimension %s.' %
                                 (inputs.name, params_shape))

            # Allocate parameters for the beta and gamma of the normalization.
            beta, gamma = None, None
            if not param_initializers:
                param_initializers = {}
            if center:
                beta_collections = utils.get_variable_collections(variables_collections,
                                                                  'beta')
                beta_initializer = param_initializers.get('beta',
                                                          init_ops.zeros_initializer())
                beta = variables.model_variable(
                    'beta',
                    shape=params_shape,
                    dtype=dtype,
                    initializer=beta_initializer,
                    collections=beta_collections,
                    trainable=trainable)
            if scale:
                gamma_collections = utils.get_variable_collections(
                    variables_collections, 'gamma')
                gamma_initializer = param_initializers.get('gamma',
                                                           init_ops.ones_initializer())
                gamma = variables.model_variable(
                    'gamma',
                    shape=params_shape,
                    dtype=dtype,
                    initializer=gamma_initializer,
                    collections=gamma_collections,
                    trainable=trainable)

            # Create moving_mean and moving_variance variables and add them to the
            # appropriate collections. We disable variable partitioning while creating
            # them, because assign_moving_average is not yet supported for partitioned
            # variables (this needs to be handled carefully, as it may break
            # the checkpoint backward compatibility).
            with variable_scope.variable_scope(
                    variable_scope.get_variable_scope()) as local_scope:
                local_scope.set_partitioner(None)
                moving_mean_collections = utils.get_variable_collections(
                    variables_collections, 'moving_mean')
                moving_mean_initializer = param_initializers.get(
                    'moving_mean', init_ops.zeros_initializer())
                moving_mean = variables.model_variable(
                    'moving_mean',
                    shape=params_shape,
                    dtype=dtype,
                    initializer=moving_mean_initializer,
                    trainable=False,
                    collections=moving_mean_collections)
                moving_variance_collections = utils.get_variable_collections(
                    variables_collections, 'moving_variance')
                moving_variance_initializer = param_initializers.get(
                    'moving_variance', init_ops.ones_initializer())
                moving_variance = variables.model_variable(
                    'moving_variance',
                    shape=params_shape,
                    dtype=dtype,
                    initializer=moving_variance_initializer,
                    trainable=False,
                    collections=moving_variance_collections)

            # If `is_training` doesn't have a constant value, because it is a `Tensor`,
            # a `Variable` or `Placeholder` then is_training_value will be None and
            # `needs_moments` will be true.
            is_training_value = utils.constant_value(is_training)
            need_moments = is_training_value is None or is_training_value
            if need_moments:
                # Calculate the moments based on the individual batch.
                if batch_weights is None:
                    if data_format == DATA_FORMAT_NCHW:
                        mean, variance = moments(inputs, moments_axes, tower_config=tower_config, keep_dims=True)
                        mean = array_ops.reshape(mean, [-1])
                        variance = array_ops.reshape(variance, [-1])
                    else:
                        mean, variance = moments(inputs, moments_axes, tower_config=tower_config)
                else:
                    if data_format == DATA_FORMAT_NCHW:
                        mean, variance = weighted_moments(
                            inputs, moments_axes, batch_weights, tower_config, keep_dims=True)
                        mean = array_ops.reshape(mean, [-1])
                        variance = array_ops.reshape(variance, [-1])
                    else:
                        mean, variance = weighted_moments(inputs, moments_axes,
                                                             batch_weights, tower_config=tower_config)

                moving_vars_fn = lambda: (moving_mean, moving_variance)
                if updates_collections is None:

                    def _force_updates():
                        """Internal function forces updates moving_vars if is_training."""
                        update_moving_mean = moving_averages.assign_moving_average(
                            moving_mean, mean, decay, zero_debias=zero_debias_moving_mean)
                        update_moving_variance = moving_averages.assign_moving_average(
                            moving_variance, variance, decay, zero_debias=False)
                        with ops.control_dependencies(
                                [update_moving_mean, update_moving_variance]):
                            return array_ops.identity(mean), array_ops.identity(variance)

                    mean, variance = utils.smart_cond(is_training, _force_updates,
                                                      moving_vars_fn)
                else:

                    def _delay_updates():
                        """Internal function that delay updates moving_vars if is_training."""
                        update_moving_mean = moving_averages.assign_moving_average(
                            moving_mean, mean, decay, zero_debias=zero_debias_moving_mean)
                        update_moving_variance = moving_averages.assign_moving_average(
                            moving_variance, variance, decay, zero_debias=False)
                        return update_moving_mean, update_moving_variance

                    update_mean, update_variance = utils.smart_cond(
                        is_training, _delay_updates, moving_vars_fn)
                    ops.add_to_collections(updates_collections, update_mean)
                    ops.add_to_collections(updates_collections, update_variance)
                    # Use computed moments during training and moving_vars otherwise.
                    vars_fn = lambda: (mean, variance)
                    mean, variance = utils.smart_cond(is_training, vars_fn, moving_vars_fn)
            else:
                mean, variance = moving_mean, moving_variance
            if data_format == DATA_FORMAT_NCHW:
                mean = array_ops.reshape(mean, params_shape_broadcast)
                variance = array_ops.reshape(variance, params_shape_broadcast)
                if beta is not None:
                    beta = array_ops.reshape(beta, params_shape_broadcast)
                if gamma is not None:
                    gamma = array_ops.reshape(gamma, params_shape_broadcast)

            # Compute batch_normalization.
            outputs = nn.batch_normalization(inputs, mean, variance, beta, gamma,
                                             epsilon)
            outputs.set_shape(inputs_shape)
            if activation_fn is not None:
                outputs = activation_fn(outputs)
            return utils.collect_named_outputs(outputs_collections, sc.name, outputs)



else:
    # None NCCL multi-gpu environment.
    def batch_norm(x, is_training=True, name="batch_norm"):
        return tf.contrib.layers.batch_norm(x,
                                            decay=0.9,
                                            updates_collections=None,
                                            epsilon=1e-5,
                                            scale=True,
                                            is_training=is_training,
                                            scope=name)
