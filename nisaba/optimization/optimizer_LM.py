import numpy as np
import scipy.optimize as sopt
import scipy.sparse.linalg as slinalg
import time

def minimize_lm(problem, num_epochs = 1,
             stepping_algorithm = 'line-search', stepping_fixed_length = 1e-1,
             mu_1 = 1.0,
             jacobian_mode = 'forward-backward',
             gmres_tol = 1e-8,
             gmres_atol = None,
             verbose = 1):

    problem.optimization_round_start('nisaba_LM')

    if stepping_algorithm == 'fixed':
        stepping_algo_code = 0
    elif stepping_algorithm == 'line-search':
        stepping_algo_code = 1
    else:
        raise Exception('Unknown stepping algorithm %s' % stepping_algorithm)

    if jacobian_mode.lower() in ['f', 'forward']:
        matrix_free = False
        forward_mode = True
    elif jacobian_mode.lower() in ['b', 'backward']:
        matrix_free = False
        forward_mode = False
    elif jacobian_mode.lower() in ['fb', 'forward-backward']:
        matrix_free = True
    else:
        raise Exception('Unknown jacobian mode %s' % jacobian_mode)

    x0 = problem.get_variables_numpy()
    E0 = problem.get_loss(x0)
    mu_ref = mu_1 * E0

    n = problem.num_variables

    I = np.eye(n)

    for _ in range(num_epochs):
        if verbose > 0:
            print('   ****************************')

        if matrix_free:

            t0 = time.time()
            F = problem.get_roots(x0)
            if verbose > 0:
                print('   ***LM roots:       %1.5f s' % (time.time() - t0))

            m = F.shape[0]
            DF = slinalg.LinearOperator((m,n),
                            matvec  = lambda vec: problem.get_jac_vec_prod(vec, x0),
                            rmatvec = lambda vec: problem.get_jacT_vec_prod(vec, x0))

            rhs = DF.T @ F
            mu = min(mu_ref, np.linalg.norm(rhs))

            def H_matvec(v):
                DF_v = problem.get_jac_vec_prod(v)
                DFT_DF_v = problem.get_jacT_vec_prod(DF_v)
                return DFT_DF_v + 0.5 * mu * v

            if verbose > 1:
                n_iter_gmres = 0
                def gmres_callback(pr_norm):
                    nonlocal n_iter_gmres
                    n_iter_gmres += 1
                    if verbose > 2:
                        print('   ****** gmres iter %d pr_norm: %1.6e' % (n_iter_gmres, pr_norm))
                gmres_callback_type = 'pr_norm'
            else:
                gmres_callback = None
                gmres_callback_type = None

            H = slinalg.LinearOperator((n,n), matvec = H_matvec)

            t0 = time.time()
            result = slinalg.gmres(H, rhs, tol = gmres_tol, atol = gmres_atol, callback = gmres_callback, callback_type = gmres_callback_type)
            if result[1] == 0:
                dx = result[0]
            elif result[1] < 0:
                raise Exception('LM method: illegal input or breakdown.')
            else:
                raise Exception('LM method: convergence to tolerance not achieved, number of iterations.')
            # result = slinalg.lsqr(DF, F)
            # dx = result[0]

            if verbose == 1:
                print('   ***LM gmres:       %1.5f s' % (time.time() - t0))
            elif verbose > 1:
                print('   ***LM gmres:       %1.5f s (%d iterations)' % (time.time() - t0, n_iter_gmres))
        else:
            t0 = time.time()
            F, DF = problem.get_roots_jac(x0, forward_mode = forward_mode)
            if verbose > 0:
                print('   ***LM roots & jac: %1.5f s' % (time.time() - t0))

            mu = min(mu_ref, np.linalg.norm(DF.T @ F))

            t0 = time.time()
            dx = np.linalg.solve(DF.T @ DF + (0.5 * mu) * I, DF.T @ F)
            if verbose > 0:
                print('   ***LM linear sys : %1.5f s' % (time.time() - t0))

        if stepping_algo_code == 0:
            x0 -= stepping_fixed_length * dx
        elif stepping_algo_code == 1:
            t0 = time.time()
            ret = sopt.line_search(problem.get_loss, problem.get_grad, x0, -dx)
            alpha = ret[0]
            if alpha is None:
                print('Levenberg-Marquardt: Line search did not converge')
                break
            if verbose > 0:
                print('   ***LM line search: %1.5f s' % (time.time() - t0))
            x0 -= alpha*dx

        if verbose > 0:
            print('   ****************************')

        problem.optimization_iter_callback()

    problem.optimization_round_end()

    return problem.history