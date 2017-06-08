import theano
import numpy as np
import scipy as sp
import theano.tensor as T 
from theano.tensor.shared_randomstreams import RandomStreams 

from theano import function, config

from scipy.optimize import minimize, fmin_bfgs, fmin_cg, fmin_l_bfgs_b

import time

import src.linalg as linalg
from src.params import *

from multiprocess import Pool
import src.multiprocess_utils as mpu

import itertools
from functools import partial

