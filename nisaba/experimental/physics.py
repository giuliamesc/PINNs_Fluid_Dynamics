import tensorflow as tf
import numpy as np

class list_style:

    @staticmethod
    def gradient_scalar(tape, v, x):
        dim = len(x)
        return  tf.stack([tape.gradient(v, x[i]) for i in range(dim)], axis = -1)

    @staticmethod
    def gradient_vector(tape, v, x):
        dim = len(x)
        # entries = [[tape.gradient(v[:, i], x[j]) for j in range(dim)] for i in range(dim)]
        # return  tf.stack([tf.stack([entries[i][j] for i in range(dim)], axis = -1) for j in range(dim)], axis = -1)
        return  tf.stack([tf.stack([tape.gradient(v[:, i], x[j]) for i in range(dim)], axis = -1) for j in range(dim)], axis = -1)

    @staticmethod
    def divergence_vector(tape, v, x):
        dim = len(x)
        return sum([tape.gradient(v[:,i], x[i]) for i in range(dim)])

    @staticmethod
    def divergence_tensor(tape, A, x):
        dim = len(x)
        return tf.stack([sum([tape.gradient(A[:,i,j], x[j]) for j in range(dim)]) for i in range(dim)], axis = -1)

    @staticmethod
    def laplacian_scalar(tape, v, x):
        dim = len(x)
        return  sum([tape.gradient(tape.gradient(v, x[i]), x[i]) for i in range(dim)])

    ##### Mechanics

    @staticmethod
    def F(tape, d, x):
        dim = len(x)
        return tf.math.add(list_style.gradient_vector(tape, d, x), np.eye(dim))

class tens_style:

    @staticmethod
    def gradient_scalar(tape, s, x):
        return tape.gradient(s, x)

    @staticmethod
    def gradient_vector(tape, v, x, d):
        # d = int(tf.shape(x)[-1])
        return  tf.stack([tape.gradient(v[:,i], x) for i in range(d)], axis = -2)

    @staticmethod
    def divergence_vector(tape, v, x, d):
        # d = int(tf.shape(x)[-1])
        # return sum([tape.gradient(v[:,i], x)[:,i] for i in range(d)])
        return tf.linalg.trace(tens_style.gradient_vector(tape, v, x, d))

    @staticmethod
    def divergence_tensor(tape, A, x, d):
        # d = int(tf.shape(x)[-1])
        return tf.stack([tens_style.divergence_vector(tape, A[:,i,:], x, d) for i in range(d)], axis = -1)

    @staticmethod
    def laplacian_scalar(tape, s, x, d):
        return tens_style.divergence_vector(tape, tape.gradient(s, x), x, d)

    @staticmethod
    def laplacian_vector(tape, v, x, d):
        return tens_style.divergence_tensor(tape, tens_style.gradient_vector(tape, v, x, d), x, d)

    ##### Mechanics

    @staticmethod
    def F(tape, u, x, d):
        # d = int(tf.shape(x)[-1])
        return tf.math.add(tens_style.gradient_vector(tape, u, x, d), np.eye(d))

    @staticmethod
    def linear_elasticity_stress(tape, u, x, mu, lam, d):
        dim = int(tf.shape(x)[-1])
        u_grad = tens_style.gradient_vector(tape, u, x, d)
        u_gradT = tf.transpose(u_grad, perm = (0, 2, 1))
        return mu * (u_grad + u_gradT) + lam * tf.expand_dims(tf.expand_dims(tf.linalg.trace(u_grad), -1), -1) * tf.expand_dims(tf.constant(np.eye(d)), 0)

def inner_vectors(a, b, d):
    # d = int(tf.shape(a)[-1]) # assuming dimensions are consistent
    return sum([a[:,i] * b[:,i] for i in range(d)])

def inner_tensors(A, B, d):
    # dim = int(tf.shape(A)[-1]) # assuming dimensions are consistent
    return sum([sum([A[:,i,j] * B[:,i,j] for j in range(d)]) for i in range(d)])

def frobenius_norm_squared(A, d):
    # dim = int(tf.shape(A)[-1]) # assuming dimensions are consistent
    return sum([sum([tf.pow(A[:,i,j], 2) for j in range(d)]) for i in range(d)])

def cofactor(A):
    J = tf.linalg.det(A)
    Ainv = tf.linalg.inv(A)
    AmT = tf.transpose(Ainv, perm = (0, 2, 1))
    return tf.expand_dims(tf.expand_dims(J, -1), 1) * AmT
