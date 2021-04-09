import tensorflow as tf
import numpy as np

class VariablesStitcher:
    """
    Helper class to reshape a list of tf.Variable's into a 1D tf.Tensor/np.array and vice versa.
    """
    def __init__(self, variables):
        """
        Parameters
        ----------
        variables
            List of variables to stitch
        """

        self.variables = variables #if isinstance(variables, (list, tuple)) else [variables]

        # obtain the shapes of the variables
        self.shapes = tf.shape_n(self.variables)
        self.n_tensors = len(self.shapes)

        count = 0
        self.idx = [] # stitch indices
        part = [] # partition indices

        for i, shape in enumerate(self.shapes):
            n = np.product(shape)
            self.idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
            part.extend([i]*n)
            count += n

        self.__num_variables = count

        self.part = tf.constant(part)

    @property
    def num_variables(self):
        """ Number or variables """
        return self.__num_variables

    def update_variables(self, params_1d):
        """
        Update the variables with a 1D object.

        Parameters
        ----------
        params_1d : 1D tf.Tensor or np.array
            Values to assign to the variables
        """

        params = tf.dynamic_partition(params_1d, self.part, self.n_tensors)
        for i, (shape, param) in enumerate(zip(self.shapes, params)):
            self.variables[i].assign(tf.reshape(param, shape))

    def reverse_stitch(self, params_1d):
        params = tf.dynamic_partition(params_1d, self.part, self.n_tensors)
        return [tf.reshape(param, shape) for i, (shape, param) in enumerate(zip(self.shapes, params))]

    def stitch(self, v = None, additional_axis = False):
        """
        Get a 1D tf.Tensor.

        Parameters
        ----------
        v : List of variables, optional
            Variables to stitch (if not passed, the variables passed to the initializer are used)
        additional_axis : bool, optional
            If True, an additional axis is expected with respect to that of the variables (used e.g. for Jacobians)
        """
        if v is None:
            v = self.variables

        if not additional_axis:
            return tf.dynamic_stitch(self.idx, v)
        else:
            return tf.transpose(tf.dynamic_stitch(self.idx, \
                      [ tf.transpose(v_i, [((i+1) % len(v_i.shape)) for i in range(len(v_i.shape))]) for v_i in v]))
