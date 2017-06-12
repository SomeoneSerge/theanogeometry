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
from src.utils import *

global dlog_likelihoodf
global theta

def bridge_sampling(lg,ltheta,dWsf,options,pars):
    (v,log_phi,seed) = pars
    theta.set_value(ltheta)
    if seed:
        srng.seed(seed)
    bridges = np.zeros((options['samples_per_obs'],n_steps.eval(),)+lg.shape)
    log_varphis = np.zeros((options['samples_per_obs'],))
    log_likelihoods = np.zeros((options['samples_per_obs'],))
    dlog_likelihoods = np.zeros((options['samples_per_obs'],)+ltheta.shape)
    global dlog_likelihoodf
    for i in range(options['samples_per_obs']):
        (ts,gsv,log_likelihood,log_varphi,dlog_likelihood) = dlog_likelihoodf(lg,v,dWsf())
        bridges[i] = gsv
        log_varphis[i] = log_varphi[-1]
        log_likelihoods[i] = log_likelihood[-1]
        dlog_likelihoods[i] = dlog_likelihood
        try:
            v = options['update_v'](v) # update v, e.g. simulate in fiber
        except KeyError:
            pass
    return (bridges,log_varphis,log_likelihoods,dlog_likelihoods,v)
# bridge(g0,A.eval(),options,(v.eval(),np.random.randint(1000)))[0].shape

# transition density
try:
    from src.quotient import *
except NameError:
    pass
def p_T(lg,v,ltheta,dWsf,bridge_sdef,phif,options,x0=None):
    vorg = v # debug
    if x0 is not None: # if lv point on manifold, lift target to fiber
        v = lift_to_fiber(v,x0)[0]
    theta.set_value(ltheta)
    bridges = np.zeros((options['samples_per_obs'],n_steps.eval(),)+lg.shape)
    log_varphis = np.zeros((options['samples_per_obs'],))
    Cgvs = np.zeros((options['samples_per_obs'],))
    for i in range(options['samples_per_obs']):
        try:
            (ts,gsv,log_likelihood,log_varphi) = bridge_sdef(lg,v,dWsf())
        except ValueError:
            print('Bridge sampling error:')
            print(v)
            print(vorg)
            print(lift_to_fiber(vorg,x0))
            print(phif(lg,v))
            raise
        bridges[i] = gsv
        log_varphis[i] = log_varphi[-1]
        Cgvs[i] = np.linalg.norm(phif(lg,v))**2
        try:
            v = options['update_v'](v) # update v, e.g. simulate in fiber
        except KeyError:
            pass
    T = 1
    return np.power(2.*np.pi*T,-.5*G_dim.eval())/np.abs(np.linalg.det(sigma.eval()))*np.mean(np.exp(-Cgvs/(2.*T))*np.exp(log_varphis))




options = {}

