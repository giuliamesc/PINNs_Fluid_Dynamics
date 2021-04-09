"""
`nisaba` library.

.. rubric:: Classes

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Loss
   LossMeanSquares
   OptimizationProblem
   VariablesStitcher


.. rubric:: Functions

.. autosummary::
   :toctree: _autosummary
   :recursive:

   minimize
   Variable
   constant


.. rubric:: Modules

"""

import tensorflow as tf

from ._version import __version__

from .optimization._minimize import minimize

from .loss import Loss, LossMeanSquares
from ._optimization_problem import OptimizationProblem
from ._tf_wrapper import GradientTape, Variable, constant
from ._variables_stitcher import VariablesStitcher

from .dataset import DataSet, DataCollection, DataEmpty

from . import optimization
from . import experimental

from . import utils
from . import config

config.set_dtype(tf.float64)