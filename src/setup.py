# # This file is part of Theano Geometry
#
# Copyright (C) 2017, Stefan Sommer (sommer@di.ku.dk)
# https://bitbucket.org/stefansommer/theanogemetry
#
# Theano Geometry is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Theano Geometry is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Theano Geometry. If not, see <http://www.gnu.org/licenses/>.
#

import theano
import numpy as np
import scipy as sp
import theano.tensor as T 
from theano.tensor.shared_randomstreams import RandomStreams 

from theano import function, config

from src.utils import *

from scipy.optimize import minimize, fmin_bfgs, fmin_cg, fmin_l_bfgs_b

import time

import src.linalg as linalg
from src.params import *

from multiprocess import Pool
import src.multiprocess_utils as mpu

import itertools
from functools import partial

