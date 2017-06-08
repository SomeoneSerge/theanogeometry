from src.setup import *
from src.manifold import *
from src.FM import *

# Development and Stochastic Development
dgamma = T.matrix()
dW = T.matrix()

## Development: (Deterministic)
def ode_Dev(dgamma,t,q):
    
    x = q[0:d]
    ui = q[d:(d+rank*d)].reshape((d,rank))

    det = T.tensordot(Hori(x,ui), dgamma, axes = [1,0])
    
    return det

dev = lambda q,dgamma: integrate(ode_Dev,q,dgamma)[1]
devf = theano.function([q,dgamma], dev(q,dgamma))

# Stochastic Development:
def sde_SD(dWt,t,q,drift):
    
    x = q[0:d]
    ui = q[d:(d+rank*d)].reshape((d,rank))

    det = T.tensordot(Hori(x,ui), drift, axes = [1,0]) # T.zeros_like(q) 
    sto = T.tensordot(Hori(x,ui), dWt, axes = [1,0])
    
    return (det, sto, T.constant(0.), T.constant(0.))

stoc_dev = lambda q,dWt,drift: integrate_sde(sde_SD,integrator_stratonovich,q,dWt,drift)[1]
stoc_devf = theano.function([q,dWt,drift], stoc_dev(q,dWt,drift))

# TO DO: Incorporate drift!!
