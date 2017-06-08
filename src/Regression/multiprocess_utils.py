# coding: utf-8

# # multiprocessing functions

## install dill and multiprocess packages

#import pip
#pip.main(['install', 'dill'])
#pip.main(['install', 'multiprocess'])

import dill
import numpy as np
from itertools import product
from functools import partial
from multiprocessing import Pool
from multiprocessing import cpu_count
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


# openPool()
# print(pool)
# closePool()
# print(pool)
# import numpy as np
# print(inputArgs(range(3),np.random.randint(1000,size=3)))



