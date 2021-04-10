"""
Utilities.
"""

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import matplotlib.transforms as transforms
except:
    pass

import tensorflow as tf
import numpy as np
import json
from . import config


def to_scalar(x):
    """
    Convert a np.array or a tf.Tensor into a scalar (float).
    """
    if isinstance(x, tf.Tensor):
        x = x.numpy()
    x = np.float64(x)
    return x

def load_json(filename):
    """
    Load a json file into a dictionary.
    """
    with open(filename) as json_file:
        return json.load(json_file)

def plot_history(history, axs = None):
    """
    Plot the training history of an Optimization Problem.

    Parameters
    ----------
    history : dict or OptimizationProblem
        History dict or Optimization Problem whose history is to be plotted.
    axs
        Matplotlib axes to plot in.
    """

    if not config.enable_graphics:
        return

    if isinstance(history, str):
        history = load_json(history)
    elif not isinstance(history, dict):
        history = history.history

    if axs is None:
        fig, axs = plt.subplots(ncols = 2, nrows = 1, figsize = (20, 8))
    else:
        fig = None

    axs[1].loglog(history['log']['iter'], history['log']['loss_global'], 'k-', linewidth = 2)

    for key, value in history['losses'].items():
        __plot_loss(history, key, value, axs, '-', '')
    for key, value in history['losses_test'].items():
        __plot_loss(history, key, value, axs, '--', '(test) ')

    for iax in range(2):
        axs[iax].legend()
        axs[iax].grid()
        axs[iax].set_xlabel('# iterations')

    tran = list()
    for iax in range(2):
        tran.append(transforms.blended_transform_factory(axs[iax].transData, axs[iax].transAxes))
    for i in range(len(history['log_rounds']['rounds'])):
        for iax in range(2):
            it = history['log_rounds']['iteration_start'][i]
            axs[iax].axvline(x = it)
            plt.text(it, 1, history['log_rounds']['rounds'][i], \
                     transform = tran[iax], \
                     bbox={'facecolor':'lightgray','alpha':0.7,'edgecolor':'black','pad':3}, \
                     ha='center', va='top',
                     rotation = 90)

    return fig

def __plot_loss(history, key, value, axs, style, prefix):
    if value['non_negative']:
        log_np = np.array(value['log'])
        axs[0].plot(history['log']['iter'], \
                    np.sqrt(log_np) if value['display_sqrt'] else log_np, \
                    style, linewidth = 1.5, \
                    label = ('%ssqrt(%s)' if value['display_sqrt'] else '%s%s') % (prefix, key))

        weight_prefix = '' if value['weight'] == 1.0 else ' * %1.2e' % value['weight']
        axs[1].plot(history['log']['iter'], \
                    value['weight'] * log_np, \
                    style, linewidth = 1.5, \
                    label = '%s%s%s' % (prefix, key, weight_prefix))

        for iax in range(2):
            axs[iax].set_xscale('symlog')
            axs[iax].set_yscale('log')

class HistoryPlotCallback():
    """Optimization Problem callback to plot real-time training history."""

    def __init__(self, frequency = 10, gui = True, filename = None, filename_history = None):
        """
        Parameters
        ----------
        frequency : int
            Refresh every <n> iterations
        gui : bool
            Toggle GUI plot (requires GUI Matplotlib backend!)
        filename : str
            If specified, the plot is saved to this file
        filename_history : str
            If specified, the history is saved to this file
        """

        self.frequency = frequency
        self.gui = gui
        self.filename = filename
        self.filename_history = filename_history
        self.__enabled = config.enable_graphics

        if self.__enabled:
            _, self.axs = plt.subplots(ncols = 2, nrows = 1, figsize = (20, 8))

    def __call__(self, pb, iter, iter_round):
        if iter % self.frequency == 0 and iter > 0:
            if self.__enabled:
                self.__draw(pb)
                if self.gui:
                    plt.draw()
                    plt.pause(1e-16)
            if self.filename_history is not None:
                pb.save_history(self.filename_history)

    def __draw(self, pb):
        for i in range(2):
            self.axs[i].cla()
        plot_history(pb, axs = self.axs)
        if self.filename is not None:
            plt.savefig(self.filename)

    def finalize(self, pb, block = False):
        """
        Show the final training history.

        Parameters
        ----------
        pb : OptimizationProblem
            Optimization Problem whose history is to be plotted.
        block : bool
            Toggle Matplotlib block.
        """
        if self.__enabled and self.gui:
            self.__draw(pb)
            plt.show(block = block)


def plot_deformed_grid(model, Wx, Wy, x0 = 0, y0 = 0, nx = 20, ny = 20, ax = None, draw = True, block = False, title = None):

    def plot_grid(x, y, ax = None, **kwargs):
        ax = ax or plt.gca()
        segs1 = np.stack((x,y), axis=2)
        segs2 = segs1.transpose(1, 0, 2)
        ax.add_collection(LineCollection(segs1, **kwargs))
        ax.add_collection(LineCollection(segs2, **kwargs))
        ax.autoscale()

    if ax is None:
        _, ax = plt.subplots()

    grid_x, grid_y = np.meshgrid(np.linspace(x0, Wx, nx), np.linspace(y0, Wy, ny))
    plot_grid(grid_x, grid_y, ax = ax, color = "lightgrey")

    grid_x_flatten = np.reshape(grid_x, (-1,))
    grid_y_flatten = np.reshape(grid_y, (-1,))

    grid = tf.stack([grid_x_flatten, grid_y_flatten], axis = -1)
    d = model(grid).numpy()

    distx = grid_x + np.reshape(d[:,0], grid_x.shape)
    disty = grid_y + np.reshape(d[:,1], grid_y.shape)

    # grid = tf.stack([grid_x,grid_y], axis = -1)
    # d = model(grid).numpy()

    # distx = grid_x + d[:,:,0]
    # disty = grid_y + d[:,:,1]

    plot_grid(distx, disty, ax = ax, color = "C0")

    if title is not None:
        ax.set_title(title)

    if draw:
        plt.draw()
        plt.pause(1e-16)
    if block:
        plt.show(block = True)

def export_ANN(ANN, filename):
    ANN_dict = dict()
    ANN_dict['layers'] = list()
    for lay in ANN.layers:
        lay_dict = dict()
        if lay.activation == tf.python.keras.activations.linear:
            lay_dict['activation'] = 'None'
        elif lay.activation == tf.nn.tanh:
            lay_dict['activation'] = 'tanh'
        else:
            raise NotImplementedError()
        lay_dict['num_neurons'] = lay.kernel.shape[1]
        lay_dict['neurons'] = list()

        for i in range(lay.kernel.shape[1]):
            neuron_dict = dict()
            if lay.bias is None:
                neuron_dict['bias'] = 0.0
            else:
                neuron_dict['bias'] = lay.bias.numpy()[i]
            neuron_dict['weights'] = list(lay.kernel.numpy()[:, i])
            lay_dict['neurons'].append(neuron_dict)

        ANN_dict['layers'].append(lay_dict)

    with open(filename, 'w') as outfile:
        json.dump(ANN_dict, outfile, indent = 2)

def import_ANN(filename):

    with open(filename) as json_file:
        ANN_dict = json.load(json_file)

    n_inputs = len(ANN_dict['layers'][0]['neurons'][0]['weights'])

    layers = list()

    for idx, lay in enumerate(ANN_dict['layers']):
        num_neurons = lay['num_neurons']
        if lay['activation'] == 'tanh':
            activation = tf.nn.tanh
        else:
            activation = None
        if idx == 0:
            input_shape = (n_inputs,)
        else:
            input_shape = ()

        layers.append(tf.keras.layers.Dense(num_neurons,
                                            input_shape = input_shape,
                                            activation = activation))

    model = tf.keras.Sequential(layers)

    for idx, lay in enumerate(ANN_dict['layers']):
        model.layers[idx].bias.assign([n['bias'] for n in lay['neurons']])
        model.layers[idx].kernel.assign(tf.transpose(tf.constant([n['weights'] for n in lay['neurons']], dtype = tf.float64)))

    return model

def load_ANN_numpy(filename):

    with open(filename) as json_file:
        ANN = json.load(json_file)

    for lay in ANN['layers']:
        lay['W'] = np.array([neuron['weights'] for neuron in lay['neurons']])
        lay['b'] = np.array([neuron['bias'] for neuron in lay['neurons']])

    def func(z):
        for lay in ANN['layers']:
            z = np.dot(lay['W'], z) + lay['b']
            if lay['activation'] == 'tanh':
                z = np.tanh(z)
        return z

    return func