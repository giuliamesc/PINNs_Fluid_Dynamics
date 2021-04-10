import numpy as np
import tensorflow as tf
import scipy.optimize as sopt
import time

def minimize(problem, method, num_epochs = None, tol = 1e-100, restart_epochs = None, options = dict(), **kwargs):


    if method == 'BFGS' and 'gtol' not in options.keys():
        options['gtol'] = 1e-100

    options['disp'] = True

    def callback(x_current):
        problem.optimization_iter_callback()
        return False

    init_params = problem.get_variables_numpy()

    problem.optimization_round_start('scipy_%s' % method)

    if restart_epochs is None:
        if num_epochs is not None:
            options['maxiter'] = num_epochs

        results = sopt.minimize(fun = problem.get_loss_grad,
                                x0 = init_params,
                                jac = True,
                                callback = callback,
                                options = options,
                                **kwargs)

    else:
        epochs_count = 0
        if num_epochs is None:
            num_epochs = np.infty
        while epochs_count < num_epochs:
            options['maxiter'] = min(num_epochs - epochs_count, restart_epochs)
            results = sopt.minimize(fun = problem.get_loss_grad,
                                    x0 = init_params,
                                    jac = True,
                                    callback = callback,
                                    options = options,
                                    **kwargs)
            epochs_count += restart_epochs

    problem.optimization_round_end()

    return results


def minimize_least_squares(problem, verbose = 2, forward_mode = True, **kwargs):

    t_init = time.time()

    results = sopt.least_squares(problem.get_roots,
                                 jac = lambda x: problem.get_jac(x, forward_mode = forward_mode),
                                 x0 = problem.get_variables_numpy(),
                                 verbose = verbose)

    print('elapsed time: %f s' % (time.time() - t_init))

    return results