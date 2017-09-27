# # This file is part of Theano Geometry
#
# Copyright (C) 2017, Stefan Sommer (sommer@di.ku.dk)
# https://bitbucket.org/stefansommer/theanogemetry
#
# Theano Geometry is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Theano Geometry is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Theano Geometry. If not, see <http://www.gnu.org/licenses/>.
#

from src.setup import *
from src.utils import *

def initialize(G,Psi=None,r=None):
    """ stochastic coadjoint motion with left/right invariant metric
    see Noise and dissipation on coadjoint orbits arXiv:1601.02249 [math.DS]
    and EulerPoincare.py """

    assert(G.invariance == 'left')

    mu = G.Vcovector() # \RR^G_dim LA cotangent vector in coordinates

    # Matrix function Psi:LA\rightarrow R^r must be defined beforehand
    # example here from arXiv:1601.02249
    if Psi is None:
        sigmaPsi = T.eye(G.dim)
        Psi = lambda mu: T.dot(sigmaPsi,mu)
        # r = Psi.shape[0]
        r = G.dim
    assert(Psi is not None and r is not None)

    assert(G.invariance == 'left')

    def sde_stochastic_coadjoint(dW,t,mu):
        xi = G.invFl(mu)
        det = -G.coad(xi,mu)
        Sigma = G.coad(mu,T.jacobian(Psi(mu),mu).dimshuffle((1,0)))
        sto = T.tensordot(Sigma,dW,(1,0))
        return (det,sto,Sigma)
    G.sde_stochastic_coadjoint = sde_stochastic_coadjoint
    G.stochastic_coadjoint = lambda mu,dWt: integrate_sde(G.sde_stochastic_coadjoint,integrator_stratonovich,mu,dWt)
    G.stochastic_coadjointf = theano.function([mu,dWt], G.stochastic_coadjoint(mu,dWt))

    # reconstruction as in Euler-Poincare / Lie-Poisson reconstruction
    if not hasattr(G,'EPrec'):
        from src.group import EulerPoincare
        EulerPoincare.initialize(G)
    G.stochastic_coadjointrec = G.EPrec
    G.stochastic_coadjointrecf = G.EPrecf

