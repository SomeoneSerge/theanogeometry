###########################
#multiprocessing functions#
###########################

import dill
import numpy as np
from itertools import product
from functools import partial
from multiprocess import Pool
from multiprocess import cpu_count

pool = None

def openPool():
    global pool
    pool = Pool(cpu_count()//2)
    
def closePool():
    global pool
    pool.terminate()
    pool = None
    
def inputArgs(*args):
    return list(zip(*args))

def getRes(res,i):
    return np.array(list(zip(*res))[i])
