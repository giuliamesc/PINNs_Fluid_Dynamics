import tensorflow as tf
import numpy as np
import time
import json
from . import _variables_stitcher
from . import _tf_wrapper as tf_wrapper
from . import utils
from . import dataset
from . import config

class OptimizationProblem:
    """
    An optimization problem.
    """
    def __init__(self, variables, losses, losses_test = [],
                 data = None,
                 callbacks = list(),
                 frequency_log = 10, frequency_print = 10,
                 print_weighted_losses = False,
                 verbosity = 1):
        """
        Parameters
        ----------
        variables : (list of) tf.Variable
            Variables to be trained.
        losses : (list of) tf.Variable
            Train loss(es).
        losses_test : Loss or list of Losses
            Test loss(es).
        frequency_log : int
            Training history is logged every <n> epochs.
        frequency_print : int
            Training history is printed to standard output every <n> epochs.
        print_weighted_losses : bool
            If true, the weighted losses are printed besides the non weighted values (only if weight is not one).
        callbacks : (list of) callable with inputs (pb, iter, iter_round)
            Callable invoked after each epoch. Inputs are:
            - `pb`: reference to this class (OptimizationProblem)
            - `iter`: global iterations counter
            - `iter_round`: iterations counter of the current optimization round
        """
        self.losses = losses if isinstance(losses, (list, tuple)) else [losses]
        self.losses_test = losses_test if isinstance(losses_test, (list, tuple)) else [losses_test]
        self.__variables = variables if isinstance(variables, (list, tuple)) else [variables]
        self.frequency_log = frequency_log
        self.frequency_print = frequency_print
        self.print_weighted_losses = print_weighted_losses
        self.callbacks = callbacks if isinstance(callbacks, (list, tuple)) else [callbacks]
        self.verbosity = verbosity
        if data is None:
            self.data = dataset.DataEmpty()
        else:
            self.data = data

        losses_names = [loss.name for loss in self.losses]
        if len(losses_names) != len(set(losses_names)):
            raise Exception('Duplicate loss name')

        losses_test_names = [loss.name for loss in self.losses_test]
        if len(losses_test_names) != len(set(losses_test_names)):
            raise Exception('Duplicate loss (test) name')

        # define variables stitcher
        self.stitcher = _variables_stitcher.VariablesStitcher(self.variables)

        # convert dtype of weights
        for loss in self.losses + self.losses_test:
            loss.set_dtype(config.get_dtype())

        self._loss_global_value = None
        self._compiled = False

        self.reset_history()
        self.initialize_functions()

    def update_loss_global(self, update_losses = True):
        """
        Update the global loss.

        Parameters
        ----------
        update_losses : bool
            Update also the losses.
        """
        if update_losses:
            for loss in self.losses:
                loss.update(self.data.current_batch)
        self._loss_global_value = tf.reduce_sum([loss.weight * loss.get() for loss in self.losses])

    @property
    def loss_global(self):
        """Get the global loss."""
        return self._loss_global_value

    @property
    def variables(self):
        """Get the trainable variables."""
        return self.__variables

    @property
    def num_variables(self):
        """Total number of scalar components of the trainable variables."""
        return self.stitcher.num_variables

    def get_variables_tensor_spec(self):
        return [tf.TensorSpec.from_tensor(tf.convert_to_tensor(v)) for v in self.variables]

    @property
    def compiled(self):
        """Flag indicating that at least one function has been compiled."""
        return self._compiled

    def compile_loss(self):
        """
        Compile the losses.
        """

        input_signature = self.data.get_tensor_spec()
        if input_signature is not None:
            input_signature = (input_signature,)
        sample_data = self.data.get_sample_element()

        for loss in self.losses + self.losses_test:
            loss.compile(input_signature, sample_data)

    def compile(self, optimizers = ['all']):
        """
        Compile the losses and the evaluation of the gradients.

        Parameters
        ----------
        optimizers : (list of) str
            Optimizers that you plan to use. The function will compile only the components used by these optimizers.
            If 'all' is passed, then all components are compiled.
        """

        optimizers = optimizers if isinstance(optimizers, (list, tuple)) else [optimizers]

        get_loss = False
        get_grad = False
        get_grads = False
        get_loss_grad = False
        get_roots = False
        get_jac = False
        get_roots_jac = False
        get_jac_vec_prod = False
        get_jacT_vec_prod = False

        if 'all' in optimizers:
            get_loss = True
            get_grad = True
            get_grads = True
            get_loss_grad = True
            get_roots = True
            get_jac = True
            get_roots_jac = True
            get_jac_vec_prod = True
            get_jacT_vec_prod = True
        if 'keras' in optimizers:
            get_loss = True
            get_grads = True
        if 'scipy' in optimizers:
            get_loss_grad = True
        if 'scipy_ls' in optimizers:
            get_roots = True
            get_jac = True
        if 'nisaba' in optimizers:
            get_loss = True
            get_grad = True
            get_roots = True
            get_roots_jac = True
            get_jac_vec_prod = True
            get_jacT_vec_prod = True
        if 'nisaba_fb' in optimizers:
            get_loss = True
            get_grad = True
            get_roots = True
            get_jac_vec_prod = True
            get_jacT_vec_prod = True

        self.compile_functions(
                losses = True,
                get_loss = get_loss,
                get_grad = get_grad,
                get_grads = get_grads,
                get_loss_grad = get_loss_grad,
                get_roots = get_roots,
                get_jac = get_jac,
                get_roots_jac = get_roots_jac,
                get_jac_vec_prod = get_jac_vec_prod,
                get_jacT_vec_prod = get_jacT_vec_prod
                )

    def initialize_functions(self):
        self.ag_loss           = self.tf_loss
        self.ag_grad           = self.tf_grad
        self.ag_loss_grad      = self.tf_loss_grad
        self.ag_grads          = self.tf_grads
        self.ag_roots          = self.tf_roots
        self.ag_jac            = self.tf_jac
        self.ag_roots_jac      = self.tf_roots_jac
        self.ag_jac_vec_prod   = self.tf_jac_vec_prod
        self.ag_jacT_vec_prod  = self.tf_jacT_vec_prod

    def compile_functions(self,
                          losses = False,
                          get_loss = False,
                          get_grad = False,
                          get_grads = False,
                          get_loss_grad = False,
                          get_roots = False,
                          get_jac = False,
                          get_roots_jac = False,
                          get_jac_vec_prod = False,
                          get_jacT_vec_prod = False):
        """
        Compile the losses and the functions for which the corresponding flag is True.
        """

        input_signature = self.data.get_tensor_spec()
        if input_signature is not None:
            input_signature = (input_signature,)
        sample_data = self.data.get_sample_element()

        if losses:
            for loss in self.losses + self.losses_test:
                loss.compile(input_signature, sample_data)

        if get_loss:
            print('Compiling tf_loss...')
            t0 = time.time()
            self.ag_loss = tf.function(self.tf_loss, input_signature = input_signature)
            self.ag_loss(sample_data)
            t_elapsed = time.time() - t0
            print('Compiling tf_loss... done! (elapsed time: %f s)' % t_elapsed)

        if get_grad:
            print('Compiling tf_grad...')
            t0 = time.time()
            self.ag_grad = tf.function(self.tf_grad, input_signature = input_signature)
            self.ag_grad(sample_data)
            t_elapsed = time.time() - t0
            print('Compiling tf_grad... done! (elapsed time: %f s)' % t_elapsed)

        if get_loss_grad:
            print('Compiling tf_loss_grad...')
            t0 = time.time()
            self.ag_loss_grad = tf.function(self.tf_loss_grad, input_signature = input_signature)
            self.ag_loss_grad(sample_data)
            t_elapsed = time.time() - t0
            print('Compiling tf_loss_grad... done! (elapsed time: %f s)' % t_elapsed)

        if get_grads:
            print('Compiling tf_grads...')
            t0 = time.time()
            self.ag_grads = tf.function(self.tf_grads, input_signature = input_signature)
            self.ag_grads(sample_data)
            t_elapsed = time.time() - t0
            print('Compiling tf_grads... done! (elapsed time: %f s)' % t_elapsed)

        if get_roots:
            print('Compiling tf_roots...')
            t0 = time.time()
            self.ag_roots = tf.function(self.tf_roots, input_signature = input_signature)
            self.ag_roots(sample_data)
            t_elapsed = time.time() - t0
            print('Compiling tf_roots... done! (elapsed time: %f s)' % t_elapsed)

        if get_jac:
            print('Compiling tf_jac...')
            t0 = time.time()
            self.ag_jac = tf.function(self.tf_jac, input_signature = input_signature)
            self.ag_jac(sample_data)
            t_elapsed = time.time() - t0
            print('Compiling tf_jac... done! (elapsed time: %f s)' % t_elapsed)

        if get_roots_jac:
            print('Compiling tf_roots_jac...')
            t0 = time.time()
            self.ag_roots_jac = tf.function(self.tf_roots_jac, input_signature = input_signature)
            self.ag_roots_jac(sample_data)
            t_elapsed = time.time() - t0
            print('Compiling tf_roots_jac... done! (elapsed time: %f s)' % t_elapsed)

        if get_jac_vec_prod:
            print('Compiling tf_jac_vec_prod...')
            t0 = time.time()

            if input_signature is not None:
                input_signature_jac_vec_prod = (self.get_variables_tensor_spec(),
                                                input_signature[0])
            else:
                input_signature_jac_vec_prod = None

            self.ag_jac_vec_prod = tf.function(self.tf_jac_vec_prod, input_signature = input_signature_jac_vec_prod)
            tangent = np.zeros(self.num_variables)
            tangent = self.stitcher.reverse_stitch(tangent)
            self.ag_jac_vec_prod(tangent, sample_data)
            t_elapsed = time.time() - t0
            print('Compiling tf_jac_vec_prod... done! (elapsed time: %f s)' % t_elapsed)

        if get_jacT_vec_prod:
            print('Compiling tf_jacT_vec_prod...')
            t0 = time.time()

            if input_signature is not None:
                input_signature_jacT_vec_prod = (tf.TensorSpec(None, dtype=config.get_dtype()),
                                                input_signature[0])
            else:
                input_signature_jacT_vec_prod = None

            self.ag_jacT_vec_prod = tf.function(self.tf_jacT_vec_prod, input_signature = input_signature_jacT_vec_prod)
            roots = self.ag_roots(sample_data)
            vec = tf.ones_like(roots)
            self.ag_jacT_vec_prod(vec, sample_data)
            t_elapsed = time.time() - t0
            print('Compiling tf_jacT_vec_prod... done! (elapsed time: %f s)' % t_elapsed)

        # if get_jac_forward_mode:
        #     print('Compiling tf_jac_forward_mode...')
        #     t0 = time.time()
        #     self.ag_jac_forward_mode = tf.function(self.tf_jac_forward_mode, input_signature = input_signature)
        #     self.ag_jac_forward_mode(sample_data)
        #     t_elapsed = time.time() - t0
        #     print('Compiling tf_jac_forward_mode... done! (elapsed time: %f s)' % t_elapsed)

        self._compiled = True

    def set_batch_size(self, batch_size):
        self.data.set_batch_size(batch_size)

    def set_epochs_per_batch(self, epochs_per_batch):
        self.data.set_epochs_per_batch(epochs_per_batch)

    def get_string_losses(self):
        """Get a formatted string with the current values of the training loss(es)."""
        return 'loss = %1.3e | %s' % (utils.to_scalar(self._loss_global_value), self.__get_string_losses(self.losses))

    def get_string_losses_test(self):
        """Get a formatted string with the current values of the test loss(es)."""
        return self.__get_string_losses(self.losses_test)

    def __get_string_losses(self, losses):
        output = ''
        for loss in losses:
           output += '%s: ' % loss.name
           if loss.display_sqrt:
              output += '%1.3e^2 ' % tf.sqrt(loss.get(tensor = False))
           else:
              output += '%1.3e ' % loss.get(tensor = False)
           if self.print_weighted_losses and not loss.weight == 1.0:
              output += '(%1.3e) ' % (loss.weight * loss.get(tensor = False))
           output += ' '
        return output

    def reset_history(self):
        """Reset the training history."""

        self.__running_optimization = False
        self.__round = 0
        self.__iter = 0

        self.history = dict()
        self.history['log'] = dict()
        self.history['log']['iter'] = list()
        self.history['log']['round'] = list()
        self.history['log']['iter_round'] = list()
        self.history['log']['loss_global'] = list()
        self.history['losses'] = dict()
        for loss in self.losses:
            self.history['losses'][loss.name] = dict()
            self.history['losses'][loss.name]['weight'] = utils.to_scalar(loss.weight)
            self.history['losses'][loss.name]['non_negative'] = loss.non_negative
            self.history['losses'][loss.name]['display_sqrt'] = loss.display_sqrt
            self.history['losses'][loss.name]['log'] = list()
        self.history['losses_test'] = dict()
        for loss in self.losses_test:
            self.history['losses_test'][loss.name] = dict()
            self.history['losses_test'][loss.name]['weight'] = utils.to_scalar(loss.weight)
            self.history['losses_test'][loss.name]['non_negative'] = loss.non_negative
            self.history['losses_test'][loss.name]['display_sqrt'] = loss.display_sqrt
            self.history['losses_test'][loss.name]['log'] = list()

        self.history['log_rounds'] = dict()
        self.history['log_rounds']['rounds'] = list()
        self.history['log_rounds']['iteration_start'] = list()

    def save_history(self, filename):
        """
        Save the training history to JSON file.

        Parameters
        ----------
        filename : str
            Output file path.
        """
        with open(filename, 'w') as outfile:
            json.dump(self.history, outfile, indent = 2)

    def optimization_round_start(self, name = None):
        """
        Callback to be used when an optimization round is started.

        Parameters
        ----------
        name : str
            Name of the optimization round.
        """
        self.__running_optimization = True
        self.__round += 1
        self.__round_name = name if name is not None else ('round %d' % self.__round)
        self.__iter_round = 0

        self.history['log_rounds']['rounds'].append(self.__round_name)
        self.history['log_rounds']['iteration_start'].append(self.__iter)

        self.data.initialize()
        self.data.next_batch()

        self.__last_time_opt_round = 0.0
        self.__last_time_loss_round = 0.0
        self.__last_time_iter_round = 0.0
        self.__time_last_call = None
        self.__time_last_call_first = None
        self.__total_time_opt_round = 0.0
        self.__total_time_loss_round = 0.0
        self.__time_round_start = time.time()
        self.optimization_iter_callback()

    def optimization_round_end(self):
        """Callback to be used when an optimization round is terminated."""

        time_round_end = time.time()

        # Update the value of the losses
        for loss in self.losses + self.losses_test:
            loss.update(self.data.current_batch)
        self.update_loss_global(update_losses = False)

        if self.verbosity > 0:
            print('=======================================')
            print('Optimization round: %s' % self.__round_name)
            print('Epochs: %d' % (self.__iter_round - 1))
            print('Losses: %s || %s' % (self.get_string_losses(), self.get_string_losses_test()))
            print('Total time: %f s' % (time_round_end - self.__time_round_start))
            print('Time (seconds):      per epoch          total')
            nit = self.__iter_round - 1
            self.__total_time_tot_round = (time_round_end - self.__time_round_start)
            self.__total_time_other_round = self.__total_time_tot_round - self.__total_time_opt_round - self.__total_time_loss_round
            print('   Training        : %1.3e %14.3f' % (self.__total_time_opt_round   / nit, self.__total_time_opt_round  ))
            print('   Printing losses : %1.3e %14.3f' % (self.__total_time_loss_round  / nit, self.__total_time_loss_round ))
            print('   Other           : %1.3e %14.3f' % (self.__total_time_other_round / nit, self.__total_time_other_round))
            print('   Total           : %1.3e %14.3f' % (self.__total_time_tot_round   / nit, self.__total_time_tot_round  ))
            print('=======================================')

        self.__running_optimization = False

    def optimization_iter_callback(self):
        """Callback to be used after each optimization round."""

        time_now = time.time()
        if self.__iter_round > 0:
            self.__last_time_opt_round = time_now - self.__time_last_call
            self.__total_time_opt_round += self.__last_time_opt_round

        if not self.__running_optimization:
            raise Exception('No optimization round started yet')

        do_log   = self.__iter_round % self.frequency_log   == 0
        do_print = self.__iter_round % self.frequency_print == 0 and self.verbosity > 0

        if do_log or do_print:
            time_loss_start = time.time()
            for loss in self.losses + self.losses_test:
                loss.update(self.data.current_batch)
            self.update_loss_global()
            if self.__iter_round > 0:
                self.__last_time_loss_round = time.time() - time_loss_start
                self.__total_time_loss_round += self.__last_time_loss_round

        if do_log:
            self.history['log']['iter'].append(self.__iter)
            self.history['log']['round'].append(self.__round)
            self.history['log']['iter_round'].append(self.__iter_round)
            self.history['log']['loss_global'].append(utils.to_scalar(self._loss_global_value))
            for loss in self.losses:
                self.history['losses'][loss.name]['log'].append(loss.get(tensor = False))
            for loss in self.losses_test:
                self.history['losses_test'][loss.name]['log'].append(loss.get(tensor = False))

        for callback in self.callbacks:
            callback(self, self.__iter, self.__iter_round)

        if self.__iter_round > 0:
            self.data.next_batch()

        time_now = time.time()
        if self.__iter_round > 0:
            self.__last_time_iter_round = time_now - self.__time_last_call
        else:
            self.__time_last_call_first = time_now
        self.__time_last_call = time_now

        if do_print:
            print('%s #%d' % (self.__round_name, self.__iter_round))
            print('          %s || %s' % (self.get_string_losses(), self.get_string_losses_test()))
            if self.__iter_round > 0:
                time_avg_opt = self.__total_time_opt_round / self.__iter_round
                time_avg_loss = self.__total_time_loss_round / self.__iter_round
                time_avg_tot = (time_now - self.__time_last_call_first) / self.__iter_round
            else:
                time_avg_opt = time_avg_loss = time_avg_tot = 0.0
            print('          epoch times (avg) : train %1.2e s | loss print %1.2e s | other %1.2e s | tot %1.2e s' % \
                (time_avg_opt, time_avg_loss, time_avg_tot - time_avg_opt - time_avg_loss, time_avg_tot))
            print('          epoch times (last): train %1.2e s | loss print %1.2e s | other %1.2e s | tot %1.2e s' % \
                (self.__last_time_opt_round, self.__last_time_loss_round, self.__last_time_iter_round - self.__last_time_opt_round - self.__last_time_loss_round, self.__last_time_iter_round))

        self.__iter += 1
        self.__iter_round += 1

    def get_variables_numpy(self):
        """Get the trainable variables converted into a static np.array."""
        return self.stitcher.stitch(self.variables).numpy()

    def tf_loss(self, data):
        """Callable returning the loss (traceable by Autograph)."""
        if self._compiled:
            print('WARNING: executing tf_loss in non compiled mode')
        return tf.reduce_sum([loss.weight * loss.call(data) for loss in self.losses])

    def get_loss(self, params_1d = None, return_numpy = True):
        """
        Callable returning the loss.

        Parameters
        ----------
        params_1d : 1D np.array
            Deseired value of the trainable. If not passed, the current trainable variables are employed.
        return_numpy : bool
            If true, the return type is np.array; if false, it is a tf.Tensor.
        """

        if params_1d is not None:
            self.stitcher.update_variables(params_1d)

        loss = self.ag_loss(self.data.current_batch)

        return loss.numpy() if return_numpy else loss

    def tf_grad(self, data):
        """Callable returning the loss gradient (traceable by Autograph)."""
        if self._compiled:
            print('WARNING: executing tf_grad in non compiled mode')

        with tf_wrapper.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.variables)
            loss_value = self.tf_loss(data)

        grads = tape.gradient(loss_value, self.variables)
        grads = self.stitcher.stitch(grads)

        return grads

    def get_grad(self, params_1d = None, return_numpy = True):
        """
        Callable returning the loss gradient.

        Parameters
        ----------
        params_1d : 1D np.array
            Deseired value of the trainable. If not passed, the current trainable variables are employed.
        return_numpy : bool
            If true, the return type is np.array; if false, it is a tf.Tensor.
        """

        if params_1d is not None:
            self.stitcher.update_variables(params_1d)

        grads = self.ag_grad(self.data.current_batch)

        return grads.numpy() if return_numpy else grads

    def tf_loss_grad(self, data):
        """Callable returning the loss  and its gradient (traceable by Autograph)."""
        if self._compiled:
            print('WARNING: executing tf_loss_grad in non compiled mode')

        with tf_wrapper.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.variables)
            loss_value = self.tf_loss(data)

        grads = tape.gradient(loss_value, self.variables)
        grads = self.stitcher.stitch(grads)

        return loss_value, grads

    def get_loss_grad(self, params_1d = None, return_numpy = True):
        """
        Callable returning the loss and its gradient.

        Parameters
        ----------
        params_1d : 1D np.array
            Deseired value of the trainable. If not passed, the current trainable variables are employed.
        return_numpy : bool
            If true, the return type is np.array; if false, it is a tf.Tensor.
        """

        if params_1d is not None:
            self.stitcher.update_variables(params_1d)

        loss_value, grads = self.ag_loss_grad(self.data.current_batch)

        if return_numpy:
            return loss_value.numpy(), grads.numpy()
        else:
            return loss_value, grads

    def tf_grads(self, data):
        if self._compiled:
            print('WARNING: executing tf_grads in non compiled mode')

        with tf_wrapper.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.variables)
            loss_value = self.tf_loss(data)

        grads = tape.gradient(loss_value, self.variables)

        return grads

    def get_grads(self, params_1d = None):

        if params_1d is not None:
            self.stitcher.update_variables(params_1d)

        return self.ag_grads(self.data.current_batch)

    def tf_roots(self, data):
        """Callable returning the roots whose sum of squares gives the loss (traceable by Autograph)."""
        if self._compiled:
            print('WARNING: executing tf_roots in non compiled mode')

        return tf.concat([loss.roots(data) * tf.sqrt(loss.weight) for loss in self.losses], 0)

    def get_roots(self, params_1d = None, return_numpy = True):
        """
        Callable returning the roots whose sum of squares gives the loss.

        Parameters
        ----------
        params_1d : 1D np.array
            Deseired value of the trainable. If not passed, the current trainable variables are employed.
        return_numpy : bool
            If true, the return type is np.array; if false, it is a tf.Tensor.
        """

        if params_1d is not None:
            self.stitcher.update_variables(params_1d)

        roots = self.ag_roots(self.data.current_batch)

        return roots.numpy() if return_numpy else roots

    def tf_jac(self, data):
        """Callable returning the the jacobian of the roots whose sum of squares gives the loss (traceable by Autograph)."""
        if self._compiled:
            print('WARNING: executing tf_jac in non compiled mode')

        with tf_wrapper.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.variables)
            roots = self.tf_roots(data)

        jac = tape.jacobian(roots, self.variables)
        jac = self.stitcher.stitch(jac, additional_axis = True)
        return jac

    def get_jac(self, params_1d = None, return_numpy = True, forward_mode = False):
        """
        Callable returning the the jacobian of the roots whose sum of squares gives the loss.

        Parameters
        ----------
        params_1d : 1D np.array
            Deseired value of the trainable. If not passed, the current trainable variables are employed.
        return_numpy : bool
            If true, the return type is np.array; if false, it is a tf.Tensor.
        """

        if params_1d is not None:
            self.stitcher.update_variables(params_1d)

        if forward_mode:
            columns = list()
            tangent = np.zeros(self.num_variables)
            for i in range(self.num_variables):
                if i > 0: tangent[i-1] = 0.0
                tangent[i] = 1.0
                columns.append(self.get_jac_vec_prod(tangent, return_numpy = return_numpy))
            if return_numpy:
                return np.stack(columns, axis = 1)
            else:
                return tf.stack(columns, axis = 1)
            # jac = self.ag_jac_forward_mode(self.data.current_batch)
            # return jac.numpy() if return_numpy else jac
        else:
            jac = self.ag_jac(self.data.current_batch)
            return jac.numpy() if return_numpy else jac

    # def tf_jac_forward_mode(self, data):

    #     self.variables_eye = tf.eye(self.num_variables, dtype = config.get_dtype()) # in the initializer

    #     jac = list()
    #     for i in range(self.num_variables):
    #         with tf.autodiff.ForwardAccumulator(
    #                             primals = self.variables,
    #                             tangents = self.stitcher.reverse_stitch(self.variables_eye[:,i])) as acc:
    #             roots = self.tf_roots(data)
    #         jac.append(acc.jvp(roots))

    #     return tf.stack(jac, axis = 1)

    # def tf_jac_forward_mode_dynamic_size(self, data):

    #     jac = tf.TensorArray(tf.float64, size = self.num_variables)
    #     for i in tf.range(self.num_variables, dtype = tf.int32):
    #         with tf.autodiff.ForwardAccumulator(
    #                             primals = self.variables,
    #                             tangents = self.stitcher.reverse_stitch(self.variables_eye[:,i])) as acc:
    #             roots = self.tf_roots(data)
    #         jac.write(i, acc.jvp(roots))

    #     return tf.transpose(jac.stack())


    def tf_jac_vec_prod(self, tangent, data):

        with tf.autodiff.ForwardAccumulator(primals = self.variables, tangents = tangent) as acc:
            roots = self.tf_roots(data)

        return acc.jvp(roots)

    def get_jac_vec_prod(self, tangent, params_1d = None, return_numpy = True):

        if params_1d is not None:
            self.stitcher.update_variables(params_1d)

        tf_tangent = self.stitcher.reverse_stitch(tangent)

        jvp = self.ag_jac_vec_prod(tf_tangent, self.data.current_batch)

        return jvp.numpy() if return_numpy else jvp

    def tf_jacT_vec_prod(self, vec, data):

        with tf_wrapper.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.variables)
            roots_times_vec = tf.reduce_sum(self.tf_roots(data) * vec)

        jTvp = tape.gradient(roots_times_vec, self.variables)
        jTvp = self.stitcher.stitch(jTvp)
        return jTvp

    def get_jacT_vec_prod(self, vec, params_1d = None, return_numpy = True):

        if params_1d is not None:
            self.stitcher.update_variables(params_1d)

        tf_vec = tf_wrapper.constant(vec)

        jTvp = self.ag_jacT_vec_prod(tf_vec, self.data.current_batch)

        return jTvp.numpy() if return_numpy else jTvp

    def tf_roots_jac(self, data):
        """Callable returning the roots whose sum of squares gives the loss and the corresponding jacobian (traceable by Autograph)."""
        if self._compiled:
            print('WARNING: executing tf_roots_jac in non compiled mode')

        with tf_wrapper.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.variables)
            roots = self.tf_roots(data)

        jac = tape.jacobian(roots, self.variables)
        jac = self.stitcher.stitch(jac, additional_axis = True)

        return roots, jac


    def get_roots_jac(self, params_1d = None, return_numpy = True, forward_mode = False):
        """
        Callable returning the roots whose sum of squares gives the loss and the corresponding jacobian.

        Parameters
        ----------
        params_1d : 1D np.array
            Deseired value of the trainable. If not passed, the current trainable variables are employed.
        return_numpy : bool
            If true, the return type is np.array; if false, it is a tf.Tensor.
        """

        if params_1d is not None:
            self.stitcher.update_variables(params_1d)

        if forward_mode:
            roots = self.get_roots(return_numpy = return_numpy)
            jac = self.get_jac(return_numpy = return_numpy, forward_mode = forward_mode)
            return roots, jac
        else:
            roots, jac = self.ag_roots_jac(self.data.current_batch)

            if return_numpy:
                return roots.numpy(), jac.numpy()
            else:
                return roots, jac

    def check_gradient(self, params_1d = None, eps_rel = 1e-6, eps_min = 1e-10):
        """
        Compare the gradient computed with Automatic Differentiation (AD) and with finite differences (FD).

        Parameters
        ----------
        params_1d : 1D np.array
            Trainable valriables in correspondence of which the gradient is computed. If not passed, the current trainable variables are employed.
        eps_rel : float
            Relative increment used in FD.
        eps_min : float
            Minimum increment used in FD.
        """

        print('Gradient check (AD vs FD)...')

        if params_1d is None:
            params_1d = self.stitcher.stitch()
        else:
            self.stitcher.update_variables(params_1d)

        if len(params_1d.shape) != 1:
            raise Exception('params_1d must have rank 1')

        num_variables = params_1d.shape[0]

        self.data.initialize()
        self.data.next_batch()

        val, grad_AD = self.get_loss_grad()

        grad_FD = np.zeros(num_variables)

        for i in range(num_variables):
            eps = max(eps_min, abs(params_1d[i])*eps_rel)
            params_1d_perturbed = np.copy(params_1d)
            params_1d_perturbed[i] += eps

            val_perturbed = self.get_loss(params_1d_perturbed)
            grad_FD[i] = ( val_perturbed - val ) / eps

        err = grad_FD - grad_AD

        err_abs_2 = np.linalg.norm(err, 2)
        err_abs_inf = np.linalg.norm(err, np.inf)
        err_rel_2 = (err_abs_2 / np.linalg.norm(grad_FD, 2))
        err_rel_inf = (err_abs_inf / np.linalg.norm(grad_FD, np.inf))
        print('difference (absolute l 2)  : %1.3e' % err_abs_2)
        print('difference (absolute l inf): %1.3e' % err_abs_inf)
        print('difference (relative l 2)  : %1.3e' % err_rel_2)
        print('difference (relative l inf): %1.3e' % err_rel_inf)

        # restore parameters
        self.stitcher.update_variables(params_1d)

        return {
            'diff_abs_2' : err_abs_2,
            'diff_abs_inf' : err_abs_inf,
            'diff_rel_2' : err_rel_2,
            'diff_rel_inf' : err_rel_inf,
             }

    def check_jacobian(self, params_1d = None, eps_rel = 1e-6, eps_min = 1e-10, forward_mode = False):
        """
        Compare the jacobian computed with Automatic Differentiation (AD) and with finite differences (FD).

        Parameters
        ----------
        params_1d : 1D np.array
            Trainable valriables in correspondence of which the jacobian is computed. If not passed, the current trainable variables are employed.
        eps_rel : float
            Relative increment used in FD.
        eps_min : float
            Minimum increment used in FD.
        """

        print('Jacobian check (AD vs FD)...')

        if params_1d is None:
            params_1d = self.stitcher.stitch()
        else:
            self.stitcher.update_variables(params_1d)

        if len(params_1d.shape) != 1:
            raise Exception('params_1d must have rank 1')

        num_variables = params_1d.shape[0]

        self.data.initialize()
        self.data.next_batch()

        val = self.get_roots()
        jac_AD = self.get_jac(forward_mode = forward_mode)

        jac_FD = np.zeros(jac_AD.shape)

        for i in range(num_variables):
            eps = max(eps_min, abs(params_1d[i])*eps_rel)
            params_1d_perturbed = np.copy(params_1d)
            params_1d_perturbed[i] += eps

            val_perturbed = self.get_roots(params_1d_perturbed)
            jac_FD[:, i] = ( val_perturbed - val ) / eps

        err = jac_FD - jac_AD

        err_abs_fro = np.linalg.norm(err, 'fro')
        err_abs_inf = abs(err).max()
        err_rel_fro = err_abs_fro / np.linalg.norm(jac_AD, 'fro')
        err_rel_inf = err_abs_inf / abs(jac_AD).max()

        print('difference (absolute fro): %1.3e' % err_abs_fro)
        print('difference (absolute inf): %1.3e' % err_abs_inf)
        print('difference (relative fro): %1.3e' % err_rel_fro)
        print('difference (relative inf): %1.3e' % err_rel_inf)

        # restore parameters
        self.stitcher.update_variables(params_1d)

        return {
            'diff_abs_fro' : err_abs_fro,
            'diff_abs_inf' : err_abs_inf,
            'diff_rel_fro' : err_rel_fro,
            'diff_rel_inf' : err_rel_inf,
             }

    def check_roots(self, params_1d = None):
        """
        Check that the sum of squares of self.get_roots() coincides with the loss.

        Parameters
        ----------
        params_1d : 1D np.array
            Trainable valriables in correspondence of which the jacobian is computed. If not passed, the current trainable variables are employed.
        """

        print('Roots check...')

        if params_1d is None:
            params_1d = self.stitcher.stitch()
        else:
            self.stitcher.update_variables(params_1d)

        if len(params_1d.shape) != 1:
            raise Exception('params_1d must have rank 1')

        self.data.initialize()
        self.data.next_batch()

        loss = self.get_loss()
        loss_roots = tf.reduce_sum(tf.square(self.get_roots()))

        err = abs(loss - loss_roots)

        print('difference: %1.3e' % err)

        # restore parameters
        self.stitcher.update_variables(params_1d)

        return err


