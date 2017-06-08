from src.manifold import *
from src.utils import *
from src.metric import *

#######################################################################
# general Brownian motion in coodinates                               #
#######################################################################

def sde_Brownian_coords(dW,t,q):
    gMsharpq = gMsharp(q)
    X = theano.tensor.slinalg.Cholesky()(gMsharpq)
    det = T.tensordot(gMsharpq,Gamma_gM(q),((0,1),(0,1)))
    sto = T.tensordot(X,dW,(1,0))
    return (det,sto,X)
Brownian_coords = lambda x,dWt: integrate_sde(sde_Brownian_coords,integrator_ito,x,dWt)
Brownian_coordsf = theano.function([q,dWt], Brownian_coords(q,dWt))

