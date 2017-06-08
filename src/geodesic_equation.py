from src.setup import *
from src.utils import *
from src.manifold import *
from src.metric import *

def ode_geodesic(t,x):
    
    dpt = - T.tensordot(T.tensordot(x[1], Gamma_gM(x[0]), axes = [0,1]),
                        x[1],axes = [1,0])
    dqt = x[1]
    
    return T.stack((dqt,dpt))

geo = lambda q,p: integrate(ode_geodesic, T.stack((q,p)))
Expteq = lambda q,p: geo(q,p)[1][:,0]
Expteqf = theano.function([q,p], Expteq(q,p))
