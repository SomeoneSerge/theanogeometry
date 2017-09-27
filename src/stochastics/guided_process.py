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

from src.utils import *
from src.linalg import *

#######################################################################
# guided processes, Delyon/Hu 2006                                    #
#######################################################################

# hit target v at time t=Tend
def get_sde_guided(sde_f,phi,sqrtCov,method='DelyonHu',integration='ito'):
    assert(integration is 'ito' or method is 'stratonovich')
    assert(method is 'DelyonHu') # more general schemes not implemented

    def sde_guided(dW,t,x,log_likelihood,log_varphi,v,*ys):
        (det,sto,X,*dys_sde) = sde_f(dW,t,x,*ys)
        h = theano.ifelse.ifelse(T.lt(t,Tend-dt/2),
                phi(x,v)/(Tend-t),
                T.zeros_like(phi(x,v))
                )
        sto = theano.ifelse.ifelse(T.lt(t,Tend-3*dt/2), # for Ito as well?

                sto,
                T.zeros_like(sto)
                )

        ### likelihood
        dW_guided = (1-.5*dt/(1-t))*dW+dt*h # for Ito as well?
        sqrtCovx = sqrtCov(x)
        Cov = dt*T.tensordot(sqrtCovx,sqrtCovx,(1,1))
        Pres =  T.nlinalg.MatrixInverse()(Cov)
        residual = T.dot(dW_guided,Pres*dW_guided)
        residual = T.tensordot(dW_guided,T.tensordot(Pres,dW_guided,(1,0)),(0,0))
        log_likelihood = .5*(-dW.shape[0]*T.log(2*np.pi)+LogAbsDet()(Pres)-residual)

        ## correction factor
        ytilde = T.tensordot(X,h*(Tend-t),1)
        tp1 = t+dt
        if integration is 'ito':
           xtp1 = x+dt*(det+T.tensordot(X,h,1))+sto
        elif integration is 'stratonovich':
           tx = x + sto
           xtp1 = x + dt*det + 0.5*(stx + sde_f(dW,tp1,tx,*ys)[1])
        Xtp1 = sde_f(dW,tp1,xtp1,*ys)[2]
        ytildetp1 = T.tensordot(Xtp1,phi(xtp1,v),1)
    
        A    = T.nlinalg.MatrixInverse()(T.tensordot(X,X,(1,1)))
        Atp1 = T.nlinalg.MatrixInverse()(T.tensordot(Xtp1,Xtp1,(1,1)))

#     at t1 term for general phi
#     dxbdxt = theano.gradient.Rop((Gx-x[0]).flatten(),x[0],dx[0]) # use this for general phi
        t2 = theano.ifelse.ifelse(T.lt(t,Tend-dt/2),
               -T.tensordot(ytilde,T.tensordot(A,dt*det,1),1)/(Tend-t),
               0.)
        t34 = theano.ifelse.ifelse(T.lt(tp1,Tend-dt/2),
               -T.tensordot(ytildetp1,T.tensordot(Atp1-A,ytildetp1,1),1)/(2*(Tend-tp1+dt*T.gt(tp1,Tend-dt/2))), # last term in divison is to avoid NaN with non-lazy Theano conditional evaluation
               0.)
        log_varphi = t2+t34

        return (det+T.tensordot(X,h,1),sto,X,log_likelihood,log_varphi,T.zeros_like(v),*dys_sde)

    return sde_guided
