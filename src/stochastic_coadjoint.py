from src.group import *
from src.utils import *
from src.energy_group import *
from src.EulerPoincare import EPrec, EPrecf

#######################################################################
#  stochastic coadjoint motion with left invariant metric             #
#######################################################################

# see Noise and dissipation on coadjoint orbits arXiv:1601.02249 [math.DS]
# and EulerPoincare.py

# Matrix function Psi:LA\rightarrow R^r must be defined beforehand
# example here from arXiv:1601.02249
sigmaPsi = T.eye(G_dim)
Psi = lambda mu: T.dot(sigmaPsi,mu)
# r = Psi.shape[0]
r = G_dim

assert(invariance == 'left')

def sde_stochastic_coadjoint(dW,t,mu):
    xi = invFl(mu)
    det = -coad(xi,mu)
    Sigma = coad(mu,T.jacobian(Psi(mu),mu).dimshuffle((1,0)))
    sto = T.tensordot(Sigma,dW,(1,0))
    return (det,sto,Sigma)
stochastic_coadjoint = lambda mu,dWt: integrate_sde(sde_stochastic_coadjoint,integrator_stratonovich,mu,dWt)
stochastic_coadjointf = theano.function([mu,dWt], stochastic_coadjoint(mu,dWt))

# reconstruction as in Euler-Poincare / Lie-Poisson reconstruction
stochastic_coadjointrec = EPrec
stochastic_coadjointrecf = EPrecf

