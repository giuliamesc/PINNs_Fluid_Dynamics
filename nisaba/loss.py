"""
Loss functions.
"""

import tensorflow as tf
import numpy as np
import time
import abc
import inspect
from . import _tf_wrapper as tf_wrapper
from . import utils
from . import config

class LossBase(abc.ABC):
    """
    Abstract class describing a generic loss.

    If you want to implement a new loss, you need to override the abstract method
    `loss_base_call`, which returns a tf.Tensor (with only one entry) representing
    the loss.
    """

    def __init__(self, name, weight, normalization, non_negative, display_sqrt):
        """
        Parameters
        ----------
        name : str
            Loss name.
        weight : scalar
            Loss weight. This scalar multiplies the current loss when it is summed
            up to other losses to give the global loss of an OptimizationProblem.
            However, the weight does not belong to the definition of the loss (i.e.
            it does not multiply the loss when it is exported in the history).
        normalization : scalar
            Loss normalization. The loss is defined as the output of loss_base_call(),
            divided by this normalization factor. Since the normalization is part of
            the definition of the loss, is is applied when the loss is exported in the
            history.
        non_negative : bool
            Flag indicating that the loss can only take nonnegative values.
        display_sqrt : bool
            If true, the square root of losse is printed instead of its absolute value (only for non-neagtive losses)
        """
        self.name = name
        self.weight = tf_wrapper.constant([weight])

        if isinstance(normalization, tf.Tensor):
            self.normalization = tf_wrapper.constant([normalization.numpy()])
        else:
            self.normalization = tf_wrapper.constant([normalization])
        self.normalization = tf.squeeze(self.normalization)
        if tf.size(self.normalization) != 1:
            raise Exception('Normalization factor should be a scalar')

        if display_sqrt and not non_negative:
            raise Exception('Only non-negative losses can be printed in sqrt version.')

        self.non_negative = non_negative
        self.display_sqrt = display_sqrt

        self.set_dtype(config.get_dtype())

        self._value = None

        self.ag_call = self.call

    def compile(self, input_signature, sample_data):
        """Compile the loss call with Autograph."""
        print('Compiling loss %s...' % self.name)

        t0 = time.time()
        self.ag_call = tf.function(self.call, input_signature = input_signature)
        result = self.ag_call(sample_data)
        t_elapsed = time.time() - t0

        if tf.size(result).numpy() != 1:
            raise Exception('The loss %s does not return a singleton.' % self.name)

        print('Compiling loss %s... done! (elapsed time: %f s)' % (self.name, t_elapsed))

    def update(self, data):
        """Update the loss value."""
        self._value = self.ag_call(data)

    def get(self, tensor = True):
        """
        Return the current value of the loss (i.e. the value after the last call to update()).

        Parameters
        ----------
        tensor : bool
            If True, a tf.Tensor is returned; if False, a scalar (float) is returned.
        """
        if tensor:
            return self._value
        else:
            return utils.to_scalar(self._value)

    def call(self, data):
        """Callable returning the loss value (after normalization)."""
        return self.loss_base_call(data) / self.normalization

    @abc.abstractmethod
    def loss_base_call(self, data):
        """Callable returning the loss value (before normalization)."""
        pass

    def roots(self, data):
        """Callable returning the roots whose sum of squares give the loss."""
        if not self.non_negative:
            raise Exception('roots are available only if non_negative = True')
        return tf.sqrt(self.loss_base_call(data) / self.normalization)

    def set_dtype(self, dtype):
        """
        Set the dtype (data type) of the loss.

        Parameters
        ----------
        dtype : tf.dtype
            dtype to set.
        """

        self.dtype = dtype
        self.weight = tf.dtypes.cast(self.weight, dtype)
        self.normalization = tf.dtypes.cast(self.normalization, dtype)

    def _repr_additional_info(self):
        return ''

    def __repr__(self):
        if self._value is None:
            val = 'None'
        else:
            val = '%1.2e' % self._val
        return '<%s \'%s\', weight = %1.2e, normalization = %1.2e, val = %s%s, at 0x%x>' % \
               (type(self).__name__,
               self.name,
               self.weight,
               self.normalization,
               val,
               self._repr_additional_info(),
               id(self))

class Loss(LossBase):
    """Loss defined by a simple call."""

    def __init__(self, name, eval_function, weight = 1.0, normalization = 1.0, non_negative = False, display_sqrt = False):
        """
        Parameters
        ----------
        name : str
            Loss name.
        eval_function : callable
            Callable returning the loss (before normalization).
        weight : scalar
            Loss weight. This scalar multiplies the current loss when it is summed
            up to other losses to give the global loss of an OptimizationProblem.
            However, the weight does not belong to the definition of the loss (i.e.
            it does not multiply the loss when it is exported in the history).
        normalization : scalar
            Loss normalization. The loss is defined as the output of loss_base_call(),
            divided by this normalization factor. Since the normalization is part of
            the definition of the loss, is is applied when the loss is exported in the
            history.
        non_negative : bool
            Flag indicating that the loss can only take nonnegative values.
        """

        num_params = len(inspect.signature(eval_function).parameters)
        if num_params == 0:
            self._eval_function = lambda data: eval_function()
        elif num_params == 1:
            self._eval_function = eval_function
        else:
            raise Exception('The signature of eval_function should have either zero or one parameters.')

        super().__init__(name, weight, normalization, non_negative, display_sqrt)

    def loss_base_call(self, data):
        return self._eval_function(data)

class LossMeanSquares(LossBase):
    """Loss defined as the mean squares of a tensor."""

    def __init__(self, name, eval_roots, weight = 1.0, normalization = 1.0):
        """
        Parameters
        ----------
        name : str
            Loss name.
        eval_roots : callable
            Callable returning the roots whose summed square give the loss (before normalization).
        weight : scalar
            Loss weight. This scalar multiplies the current loss when it is summed
            up to other losses to give the global loss of an OptimizationProblem.
            However, the weight does not belong to the definition of the loss (i.e.
            it does not multiply the loss when it is exported in the history).
        normalization : scalar
            Loss normalization. The loss is defined as the output of loss_base_call(),
            divided by this normalization factor. Since the normalization is part of
            the definition of the loss, is is applied when the loss is exported in the
            history.
        """

        num_params = len(inspect.signature(eval_roots).parameters)
        if num_params == 0:
            self._eval_roots = lambda data: eval_roots()

            roots = eval_roots()
            if isinstance(roots, (list, tuple)):
                n_squares = len(roots)
            elif isinstance(roots, (tf.Tensor, tf.Variable)):
                n_squares = tf.size(roots).numpy()
            else:
                raise Exception('The function eval_roots can only return: list, tuple, tf.Tensor, tf.Variable.')
            self.n_squares = tf.constant([n_squares], dtype = tf.float32)

        elif num_params == 1:
            self._eval_roots = eval_roots
            self.n_squares = None # depends on batch size
        else:
            raise Exception('The signature of eval_roots should have either zero or one parameters.')

        super().__init__(name, weight, normalization, True, True)

    def loss_base_call(self, data):
        return tf.reduce_mean(tf.square(self._eval_roots(data)))

    def roots(self, data):
        roots = self._eval_roots(data)
        n_squares = tf.cast(tf.reduce_prod(tf.shape(roots)), self.dtype)
        return tf.reshape(roots, (-1,)) / tf.sqrt( n_squares * self.normalization )

    def set_dtype(self, dtype):
        super().set_dtype(dtype)
        if self.n_squares is not None:
            self.n_squares = tf.dtypes.cast(self.n_squares, dtype)

    def _repr_additional_info(self):
        if self.n_squares is not None:
            return ', n_squares = %d' % self.n_squares
        else:
            return ''