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

###############################################################
# Langevin equations https://arxiv.org/abs/1605.09276
###############################################################
def initialize(M):
    q = M.coords()
    p = M.coordscovector()
    
    l = T.scalar()
    s = T.scalar()

    dW = M.element()

    dq = lambda q,p: T.grad(M.H(q,p),p)
    dp = lambda q,p: -T.grad(M.H(q,p),q)

    def sde_Langevin(dW,t,x,l,s):
        dqt = dq(x[0],x[1])
        dpt = -l*dq(x[0],x[1])+dp(x[0],x[1])

        X = T.stack((T.zeros((M.dim,M.dim)),s*T.eye(M.dim)))
        det = T.stack((dqt,dpt))
        sto = T.tensordot(X,dW,(1,0))
        return (det,sto,X,l,s)
    M.Langevin_qp = lambda q,p,l,s,dWt: integrate_sde(sde_Langevin,integrator_ito,T.stack((q,p)),dWt,l,s)
    M.Langevin_qpf = theano.function([q,p,l,s,dWt], M.Langevin_qp(q,p,l,s,dWt))

    M.Langevin = lambda q,p,l,s,dWt: M.Langevin_qp(q,p,l,s,dWt)[0:2]
    M.Langevinf = theano.function([q,p,l,s,dWt], M.Langevin(q,p,l,s,dWt))
