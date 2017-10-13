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
from src.params import *
#try:
#    from src.manifold import *
#except NameError:
#    pass
#try:
#    from src.group import *
#except NameError:
#    pass

#######################################################################
# various useful functions                                            #
#######################################################################

def constant(c):
    """ return Theano constant with value of parameter c """
    try:
        return T.constant(c)
    except:
        return c

# numeric optimizer
def get_minimizer(f,method=None,options=None):
    x = T.vector()
    ff = theano.function([x],f(x))
    gradff = theano.function([x],T.grad(f(x),x))

    def fopt(x):
        return (ff(x),gradff(x))

    return (lambda x0: minimize(fopt, x0, method=method, jac=True, options=None))

# Integrator (non-stochastic)
def integrator(ode_f,method=default_method):

    # euler:
    def euler(*y):
        t = y[-2]
        x = y[-1]
        return (t+dt,x+dt*ode_f(*y))

    # Runge-kutta:
    def integrator_rk4(ode_f):
        def rk4(*y):
            t = y[-2]
            x = y[-1]
            k1 = ode_f(y[0:-2],t,x)
            k2 = ode_f(y[0:-2],t+dt/2,x + dt/2*k1)
            k3 = ode_f(y[0:-2],t+dt/2,x + dt/2*k2)
            k4 = ode_f(y[0:-2],t,x + dt*k3)
            return (t+dt,x + dt/6*(k1 + 2*k2 + 2*k3 + k4))

    if method == 'euler':
        return euler
    elif method == 'rk4':
        return rk4
    else:
        assert(False)

# return symbolic path given ode and integrator
def integrate(ode,x,*y):
    (cout, updates) = theano.scan(fn=integrator(ode),
            outputs_info=[T.constant(0.),x],
            sequences=[*y],
            n_steps=n_steps)
    return cout

# sde functions should return (det,sto,Sigma) where
# det is determinisitc part, sto is stochastic part,
# and Sigma stochastic generator (i.e. often sto=dot(Sigma,dW)

# standard noise realisations
srng = RandomStreams()#seed=42)
dWt = T.matrix() # n_steps x d, d usually manifold dimension
dWs = lambda d: srng.normal((n_steps,d), std=np.sqrt(dt))
d = T.scalar(dtype='int64')
dWsf = theano.function([d],dWs(d))
del d

def integrator_stratonovich(sde_f):
    def euler_heun(dW,t,x,*ys):
        (detx, stox, X, *dys) = sde_f(dW,t,x,*ys)
        tx = x + stox
        ys_new = ()
        for (y,dy) in zip(ys,dys):
            ys_new = ys_new + (y+dt*dy,)
        return (t+dt,x + dt*detx + 0.5*(stox + sde_f(dW,t+dt,tx,*ys)[1]), *ys_new)

    return euler_heun

def integrator_ito(sde_f):
    def euler(dW,t,x,*ys):
        (detx, stox, X, *dys) = sde_f(dW,t,x,*ys)
        ys_new = ()
        for (y,dy) in zip(ys,dys):
            ys_new = ys_new + (y+dt*dy,)
        return (t+dt,x + dt*detx + stox, *ys_new)

    return euler

def integrate_sde(sde,integrator,x,dWt,*ys):
    (cout, updates) = theano.scan(fn=integrator(sde),
            outputs_info=[T.constant(0.),x, *ys],
            sequences=[dWt],
            n_steps=n_steps)
    return cout

## Gram-Schmidt:
#def GramSchmidt_f(innerProd):
#    def GS(Frame,q):
#
#        if len(Frame.shape) == 1:
#            gS = Frame/np.sqrt(innerProd(Frame,Frame,q))
#        else:
#            gS = np.zeros_like(Frame)
#            for j in range(0,Frame.shape[1]):
#                gS[:,j] = Frame[:,j]
#                for i in range(0,j):
#                    foo = innerProd(Frame[:,j],gS[:,i],q)/ innerProd(gS[:,i],gS[:,i],q)
#                    gS[:,j] = gS[:,j] - foo*gS[:,i]
#
#                gS[:,j] = gS[:,j]/np.sqrt(innerProd(gS[:,j],gS[:,j],q))
#
#        return gS
#
#    return GS



