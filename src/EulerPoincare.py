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

from src.group import *
from src.utils import *
from src.energy_group import *

###############################################################
# geodesic integration, Euler-Poincare form                   #
###############################################################

assert(invariance == 'left')

def ode_EP(t,mu):
    xi = invFl(mu)
    dmut = -coad(xi,mu)
    return dmut
EP = lambda mu: integrate(ode_EP,mu)
EPf = theano.function([mu], EP(mu))

# reconstruction
def ode_EPrec(mu,t,g):
    xi = invFl(mu)
    dgt = dL(g,e,VtoLA(xi))
    return dgt
EPrec = lambda g,mus: integrate(ode_EPrec,g,mus)
mus = T.matrix() # mu for each time step
EPrecf = theano.function([g,mus], EPrec(g,mus)) 

### geodesics
coExp = lambda g,mu: EPrec(g,EP(mu)[1])[1][-1]
Exp = lambda g,v: coExp(g,flatV(v))
coExpt = lambda g,mu: EPrec(g,EP(mu)[1])
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
