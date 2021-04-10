import numpy as np
import tensorflow as tf
import time

def minimize(problem, optimizer, num_epochs = 1):

    problem.optimization_round_start('keras_%s' % type(optimizer).__name__)
    for _ in range(num_epochs):
        # optimizer.minimize(lambda: problem.get_loss(return_numpy = False), problem.variables)
        optimizer.apply_gradients(zip(problem.get_grads(), problem.variables))
        problem.optimization_iter_callback()
    problem.optimization_round_end()

    return problem.history