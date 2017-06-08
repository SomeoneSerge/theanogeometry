from src.setup import *
from src.utils import *
from src.manifold import *
from src.metric import *

# Curvature:
def R(x):
    return T.tensordot(Gamma_gM(x),Gamma_gM(x),axes = [0,2]).dimshuffle(0,3,1,2) - T.tensordot(Gamma_gM(x),Gamma_gM(x),axes = [0,2]).dimshuffle(3,0,1,2) + T.jacobian(Gamma_gM(x).flatten(),x).reshape((d,d,d,d)).dimshuffle(1,3,2,0) - T.jacobian(Gamma_gM(x).flatten(),x).reshape((d,d,d,d)).dimshuffle(3,1,2,0)

def R_ui(x,ui):
    return T.tensordot(T.nlinalg.matrix_inverse(ui),T.tensordot(R(x),ui,(2,0)),(1,2)).dimshuffle(1,2,0,3)

Rf = theano.function([x], R(x))
R_uif = theano.function([x,ui], R_ui(x,ui))

# Sectional Curvature:


