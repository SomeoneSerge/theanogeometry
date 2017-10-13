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

# hit target v at time t=Tend
def get_sde_fiber(sde_f,proj,G,M):
    def sde_fiber(dW,t,x,*ys):
        (det,sto,X,*dys_sde) = sde_f(dW,t,x,*ys)
        
        # compute kernel of proj derivative with respect to inv A metric
        rank = M.dim
        Xframe = T.tensordot(G.invpf(x,G.eiLA),G.sigma,(2,0))
        Xframe_inv = T.nlinalg.MatrixPinv()(Xframe.reshape((-1,G.dim)))
        dproj = T.tensordot(T.jacobian(proj(x).flatten(),x),
                            Xframe,
                           ((1,2),(0,1))).reshape((proj(x).shape[0],G.dim))
        (_,_,Vh) = T.nlinalg.svd(dproj,full_matrices=True)
        ns = Vh[rank:].T # null space
        proj_ns = T.tensordot(ns,ns,(1,1))
        
        det = T.tensordot(Xframe,T.tensordot(proj_ns,T.tensordot(Xframe_inv,det.flatten(),(1,0)),(1,0)),(2,0)).reshape(x.shape)
        sto = T.tensordot(Xframe,T.tensordot(proj_ns,T.tensordot(Xframe_inv,sto.flatten(),(1,0)),(1,0)),(2,0)).reshape(x.shape)
        X = T.tensordot(Xframe,T.tensordot(proj_ns,T.tensordot(Xframe_inv,X.flatten(),(1,0)),(1,0)),(2,0)).reshape(X.shape)
        
        return (det,sto,X,*dys_sde)

    return sde_fiber

# find g in fiber above x closests to g0
from scipy.optimize import minimize
def lift_to_fiber(x,x0,G,M):
    shoot = lambda hatxi: G.gVf(hatxi,hatxi)
    try:
        hatxi = minimize(shoot,
                np.zeros(G.dim.eval()),
                method='COBYLA',
                constraints={'type':'ineq','fun':lambda hatxi: np.min((G.injectivity_radius.eval()-np.max(hatxi),
                                                                      1e-8-np.linalg.norm(M.actf(G.expf(G.VtoLAf(hatxi)),x0)-x)**2))},
                ).x
    except NameError: # injectivity radius not defined
        hatxi = minimize(shoot,np.zeros(G.dim.eval())).x
    l0 = G.expf(G.VtoLAf(hatxi))
    try: # project to group if to_group function is available
        l0 = G.to_groupf(l0)
    except NameError:
        pass
    return (l0,hatxi)

# estimate fiber volume
from multiprocess import cpu_count
import scipy.special
from src.plotting import *

def fiber_samples(G,Brownian_fiberf,L,pars):
    (seed,) = pars
    if seed:
        srng.seed(seed)
    gsl = np.zeros((L,) + G.e.eval().shape)
    dsl = np.zeros(L)
    (ts, gs) = Brownian_fiberf(G.e.eval(), dWsf(G.dim.eval()))
    vl = gs[-1]  # starting point
    for l in range(L):
        (ts, gs) = Brownian_fiberf(vl, dWsf(G.dim.eval()))
        gsl[l] = gs[-1]
        dsl[l] = np.linalg.norm(G.LAtoVf(G.logf(gs[-1])))  # distance to sample with canonical biinvariant metric
        vl = gs[-1]

    return (gsl, dsl)

def estimate_fiber_volume(G, M, lfiber_samples, nr_samples=100, plot_dist_histogram=False, plot_samples=False):
    """ estimate fiber volume with restricted Riemannian G volume element (biinvariant metric) """
    L = nr_samples // (cpu_count() // 2)  # samples per processor

    try:
        mpu.openPool()
        sol = mpu.pool.imap(partial(lfiber_samples, L), mpu.inputArgs(np.random.randint(1000, size=cpu_count() // 2)))
        res = list(sol)
        gsl = mpu.getRes(res, 0).reshape((-1,) + G.e.eval().shape)
        dsl = mpu.getRes(res, 1).flatten()
    except:
        mpu.closePool()
        raise
    else:
        mpu.closePool()

    if plot_dist_histogram:
        # distance histogram
        plt.hist(dsl, 20)

    if plot_samples:
        # plot samples
        newfig()
        for l in range(0, L):
            G.plotg(gsl[l])
        plt.show()

    # count percentage of samples below distance d to e relative to volume of d-ball
    d = np.max(dsl)  # distance must be smaller than fiber radius
    fiber_dim = G.dim.eval() - M.dim.eval()
    ball_volume = np.pi ** (fiber_dim / 2) / scipy.special.gamma(fiber_dim / 2 + 1) * d ** fiber_dim

    return ball_volume / (np.sum(dsl < d) / (dsl.size))
