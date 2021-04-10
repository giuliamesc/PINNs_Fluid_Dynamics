from . import optimizer_scipy
from . import optimizer_keras
from . import optimizer_LM

def minimize(problem, backend, method, *args, **kwargs):
    """
    Minimize the loss associated with an OptimizationProblem.

    Parameters
    ----------
    problem : OptimizationProblem
        Problem whose loss and whose variables are considered
    backend : str
        Backend
    method
        Optimization method
    """
    if backend.lower() == 'scipy':
        return optimizer_scipy.minimize(problem, method = method, *args, **kwargs)

    elif backend.lower() == 'scipy_ls':
        return optimizer_scipy.minimize_least_squares(problem, method = method, *args, **kwargs)

    elif backend.lower() == 'keras':
        return optimizer_keras.minimize(problem, method, *args, **kwargs)

    elif backend.lower() == 'nisaba':
        if method.lower() == 'lm':
            return optimizer_LM.minimize_lm(problem, *args, **kwargs)
        else:
            raise Exception('Unknown method: %s' % method)

    else:
        raise Exception('Unknown backend: %s' % backend)

