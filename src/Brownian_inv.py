from src.group import *
from src.utils import *
from src.metric import *

#######################################################################
# general Brownian motion with respect to left/right invariant metric #
#######################################################################

assert(invariance == 'left')

def sde_Brownian_inv(dW,t,g):
    X = T.tensordot(invpf(g,eiLA),sigma,(2,0))
    det = -.5*T.tensordot(T.diagonal(C,0,2).sum(1),X,(0,2))
    sto = T.tensordot(X,dW,(2,0))
    return (det,sto,X)
Brownian_inv = lambda g,dWt: integrate_sde(sde_Brownian_inv,integrator_stratonovich,g,dWt)
Brownian_invf = theano.function([g,dWt], Brownian_inv(g,dWt))

