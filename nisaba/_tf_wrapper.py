import tensorflow as tf
from . import config

class GradientTape(tf.GradientTape):
    """
    Wrapper to tf.GradientTape with default unconnected_gradients = tf.UnconnectedGradients.ZERO
    """

    def gradient(self, y, x, unconnected_gradients = tf.UnconnectedGradients.ZERO, *args, **kwargs):
        return super().gradient(y, x, unconnected_gradients = unconnected_gradients, *args, **kwargs)

    def jacobian(self, y, x, unconnected_gradients = tf.UnconnectedGradients.ZERO, *args, **kwargs):
        return super().jacobian(y, x, unconnected_gradients = unconnected_gradients, *args, **kwargs)

def Variable(initial_value = None, dtype = config.__dtype, *args, **kwargs):
    """
    Wrapper to tf.Variable with default dtype set equal to nisaba.config.dtype
    """
    return tf.Variable(initial_value, dtype = dtype, *args, **kwargs)

def constant(value, dtype = config.__dtype, *args, **kwargs):
    """
    Wrapper to tf.constant with default dtype set equal to nisaba.config.dtype
    """
    return tf.constant(value, dtype = dtype, *args, **kwargs)
