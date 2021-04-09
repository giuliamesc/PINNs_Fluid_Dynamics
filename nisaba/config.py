"""
Configuration.

.. data:: enable_graphics

    Globally toggle the generation of live graphics.

"""

import tensorflow as tf

__dtype = tf.float64

def set_dtype(dtype):
    """Set the default dtype."""
    global __dtype

    __dtype = dtype

    if dtype == tf.float16:
        tf.keras.backend.set_floatx('float16')
    if dtype == tf.float32:
        tf.keras.backend.set_floatx('float32')
    if dtype == tf.float64:
        tf.keras.backend.set_floatx('float64')

def get_dtype():
    """Get the default dtype."""
    global __dtype
    return __dtype

enable_graphics = True
