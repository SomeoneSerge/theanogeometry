from src.group import *
from src.utils import *
from src.energy_group import *

###############################################################
# geodesic integration, Lie-Poission form                     #
###############################################################

assert(invariance == 'left')

def ode_LP(t,mu):
    dmut = coad(dHminusdmu(mu),mu)
    return dmut
LP = lambda mu: integrate(ode_LP,mu)
LPf = theano.function([mu], LP(mu))

# reconstruction
def ode_LPrec(mu,t,g):
    dgt = dL(g,e,VtoLA(dHminusdmu(mu)))
    return dgt
LPrec = lambda g,mus: integrate(ode_LPrec,g,mus)
mus = T.matrix() # mu for each time step
LPrecf = theano.function([g,mus], LPrec(g,mus)) 

### geodesics
coExp = lambda g,mu: LPrec(g,LP(mu)[1])[1][-1]
Exp = lambda g,v: coExp(g,flatV(v))
coExpt = lambda g,mu: LPrec(g,LP(mu)[1])
Expt = lambda g,v: coExpt(g,flatV(v))
DcoExp = lambda g,mu: (
    T.jacobian(coExp(g,mu).flatten(),g).reshape(N,N,N,N),
    T.jacobian(coExp(g,mu).flatten(),mu).reshape(N,N,G_dim)
    )
#loss = 1./G_emb_dim*T.sum(T.sqr(Exp(g,mu)-h))
#dloss = (T.grad(loss,g),T.grad(loss,g))
Expf = theano.function([g,v], Exp(g,v))
Exptf = theano.function([g,v], Expt(g,v))
coExpf = theano.function([g,mu], coExp(g,mu))
coExptf = theano.function([g,mu], coExpt(g,mu))
#lossf = theano.function([g,mu,h], loss)
#dlossf = theano.function([g,mu,h], [loss, dloss[0], dloss[1]])
