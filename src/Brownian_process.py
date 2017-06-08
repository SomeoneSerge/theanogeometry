from src.group import *
from src.metric import *
from src.utils import *

#######################################################################
# Brownian process with respect to left/right invariant metric        #
#######################################################################

assert(invariance == 'left')

def sde_Brownian_process(dW,t,g):
    X = T.tensordot(invpf(g,eiLA),sigma,(2,0))
    det = T.zeros_like(g)
    sto = T.tensordot(X,dW,(2,0))
    return (det,sto,X)
Brownian_process = lambda g,dWt: integrate_sde(sde_Brownian_process,integrator_stratonovich,g,dWt)
Brownian_processf = theano.function([g,dWt], Brownian_process(g,dWt))
