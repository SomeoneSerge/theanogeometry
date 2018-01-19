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
def get_sde_guided(sde_f, phi, sqrtCov, A=None, method='DelyonHu', integration='ito'):
    assert (integration is 'ito' or integration is 'stratonovich')
    assert (method is 'DelyonHu')  # more general schemes not implemented

    def sde_guided(dW, t, x, log_likelihood, log_varphi, h, v, *ys):
        (det, sto, X, *dys_sde) = sde_f(dW, t, x, *ys)
        h = theano.ifelse.ifelse(T.lt(t, Tend - dt / 2),
                                 phi(x, v) / (Tend - t),
                                 T.zeros_like(phi(x, v))
                                 )
        sto = theano.ifelse.ifelse(T.lt(t, Tend - 3 * dt / 2),  # for Ito as well?
                                   sto,
                                   T.zeros_like(sto)
                                   )

        ### likelihood
        dW_guided = (1 - .5 * dt / (1 - t)) * dW + dt * h  # for Ito as well?
        sqrtCovx = sqrtCov(x)
        Cov = dt * T.tensordot(sqrtCovx, sqrtCovx, (1, 1))
        Pres = T.nlinalg.MatrixInverse()(Cov)
        residual = T.tensordot(dW_guided, T.tensordot(Pres, dW_guided, (1, 0)), (0, 0))
        log_likelihood = .5 * (-dW.shape[0] * T.log(2 * np.pi) + LogAbsDet()(Pres) - residual)

        ## correction factor
        ytilde = T.tensordot(X, h * (Tend - t), 1)
        tp1 = t + dt
        if integration is 'ito':
            xtp1 = x + dt * (det + T.tensordot(X, h, 1)) + sto
        elif integration is 'stratonovich':
            tx = x + sto
            xtp1 = x + dt * det + 0.5 * (sto + sde_f(dW, tp1, tx, *ys)[1])
        Xtp1 = sde_f(dW, tp1, xtp1, *ys)[2]
        ytildetp1 = T.tensordot(Xtp1, phi(xtp1, v), 1)

        # set default A if not specified
        Af = A if A is not None else lambda x, v, w: T.tensordot(v, T.tensordot(T.nlinalg.MatrixInverse()(T.tensordot(X, X, (1, 1))), w, 1), 1)

        #     add t1 term for general phi
        #     dxbdxt = theano.gradient.Rop((Gx-x[0]).flatten(),x[0],dx[0]) # use this for general phi
        t2 = theano.ifelse.ifelse(T.lt(t, Tend - dt / 2),
                                  -Af(x, ytilde, dt * det) / (Tend - t),
                                  # check det term for Stratonovich (correction likely missing)
                                  0.)
        t34 = theano.ifelse.ifelse(T.lt(tp1, Tend - dt / 2),
                                   -(Af(xtp1, ytildetp1, ytildetp1) - Af(x, ytildetp1, ytildetp1)) / (
                                   2 * (Tend - tp1 + dt * T.gt(tp1, Tend - dt / 2))),
                                   # last term in divison is to avoid NaN with non-lazy Theano conditional evaluation
                                   0.)
        log_varphi = t2 + t34

        return (det + T.tensordot(X, h, 1), sto, X, log_likelihood, log_varphi, dW_guided/dt, T.zeros_like(v), *dys_sde)

    return sde_guided

def get_guided_likelihood(M, sde_f, phi, sqrtCov, q, thetas, A=None, method='DelyonHu', integration='ito'):
    sde_guided = get_sde_guided(sde_f, phi, sqrtCov, A, method, integration)
    guided = lambda q, v, dWt: integrate_sde(sde_guided,
                                             integrator_ito if method is 'ito' else integrator_stratonovich,
                                             q, dWt, T.constant(0.), T.constant(0.), T.zeros_like(dWt[0]), v)
    v = M.element()
    guidedf = theano.function([q, v, dWt], guided(q, v, dWt))

    # derivatives
    def dlog_likelihood(q, v, dWt):
        s = guided(q, v, dWt)[:4]
        dlog_likelihoods = tuple(T.grad(s[2][-1], theta) for theta in thetas)

        return tuple(s) + dlog_likelihoods

    dlog_likelihoodf = theano.function([q, v, dWt], dlog_likelihood(q, v, dWt))

    return (dlog_likelihood, dlog_likelihoodf, guided, guidedf)

def bridge_sampling(lg,bridge_sdef,dWsf,options,pars):
    """ sample samples_per_obs bridges """
    (v,seed) = pars
    if seed:
        srng.seed(seed)
    bridges = np.zeros((options['samples_per_obs'],n_steps.eval(),)+lg.shape)
    log_varphis = np.zeros((options['samples_per_obs'],))
    log_likelihoods = np.zeros((options['samples_per_obs'],))
    for i in range(options['samples_per_obs']):
        (ts,gs,log_likelihood,log_varphi) = bridge_sdef(lg,v,dWsf())[:4]
        bridges[i] = gs
        log_varphis[i] = log_varphi[-1]
        log_likelihoods[i] = log_likelihood[-1]
        try:
            v = options['update_vf'](v) # update v, e.g. simulate in fiber
        except KeyError:
            pass
    return (bridges,log_varphis,log_likelihoods,v)

# helper for log-transition density
def p_T_log_p_T(g, v, dWs, bridge_sde, phi, options, sigma=None, sde=None):
    """ Monte Carlo approximation of log transition density from guided process """
    if sigma is None and sde is not None:
        (_, _, XT) = sde(dWs, Tend, v)  # starting point of SDE, we need diffusion field X at t=0
        sigma = XT
    assert (sigma is not None)
    
    # sample noise
    (cout, updates) = theano.scan(fn=lambda x: dWs,
                                  outputs_info=[T.zeros_like(dWs)],
                                  n_steps=options['samples_per_obs'])
    dWsi = cout

    if not 'update_v' in options:
        # v constant throughout sampling
        print("transition density with v constant")

        # bridges
        Cgv = T.sum(phi(g, v) ** 2)
        def bridge_logvarphis(dWs, log_varphi):
            (ts, gs, log_likelihood, log_varphi) = bridge_sde(g, v, dWs)[:4]
            return log_varphi[-1]

        (cout, updates) = theano.scan(fn=bridge_logvarphis,
                                      outputs_info=[T.constant(0.)],
                                      sequences=[dWsi])
        log_varphi = T.log(T.mean(T.exp(cout)))
        log_p_T = -.5 * sigma.shape[0] * T.log(2. * np.pi * Tend) - LogAbsDet()(sigma) - Cgv / (2. * Tend) + log_varphi
        p_T = T.exp(log_p_T)
    else:
        # update v during sampling, e.g. for fiber densities
        print("transition density with v updates")

        # bridges
        def bridge_p_T(dWs, lp_T, lv):
            Cgv = T.sum(phi(g, lv) ** 2)
            (ts, gs, log_likelihood, log_varphi, _) = bridge_sde(g, lv, dWs)            
            lp_T =  T.power(2.*np.pi*Tend,-.5*sigma.shape[0])/T.abs_(T.nlinalg.Det()(sigma))*T.exp(-Cgv/(2.*Tend))*T.exp(log_varphi[-1])
            lv = options['update_v'](lv)                        
            return (lp_T, lv)

        (cout, updates) = theano.scan(fn=bridge_p_T,
                                      outputs_info=[T.constant(0.), v],
                                      sequences=[dWsi])
        p_T = T.mean(cout[:][0])
        log_p_T = T.log(p_T)
        v = cout[-1][1]
    
    return (p_T,log_p_T,v)

def p_T(*args,**kwargs): return p_T_log_p_T(*args,**kwargs)[0]
def log_p_T(*args,**kwargs): return p_T_log_p_T(*args,**kwargs)[1]

def dp_T(thetas,*args,**kwargs):
    """ Monte Carlo approximation of transition density gradient """
    lp_T = p_T(*args,**kwargs)
    return (lp_T,)+tuple(T.grad(lp_T,theta) for theta in thetas)

def dlog_p_T(thetas,*args,**kwargs):
    """ Monte Carlo approximation of log transition density gradient """
    llog_p_T = log_p_T(*args,**kwargs)
    return (llog_p_T,)+tuple(T.grad(llog_p_T,theta) for theta in thetas)

#def log_p_T_numeric(lg,v,dWsf,bridge_sdef,phif,options,sigma=None,sdef=None,x0=None):
#    """ numpy version of Monte Carlo log transition density """
#    vorg = v # debug
#    if x0 is not None: # if lv point on manifold, lift target to fiber
#        v = lift_to_fiber(v,x0)[0]
#    if sdef is not None:
#        (_,_,XT) = sdef(Tend.eval(),v) # starting point of SDE, we need diffusion field X at t=0
#        sigma = XT
#    elif sigma is not None:
#        sigma = sigma.eval()
#    assert(sigma is not None)
#    bridges = np.zeros((options['samples_per_obs'],n_steps.eval(),)+lg.shape)
#    log_varphis = np.zeros((options['samples_per_obs'],))
#    Cgvs = np.zeros((options['samples_per_obs'],))
#    for i in range(options['samples_per_obs']):
#        try:
#            (ts,gs,log_likelihood,log_varphi) = bridge_sdef(lg,v,dWsf())
#        except ValueError:
#            print('Bridge sampling error:')
#            print(v)
#            print(vorg)
#            print(lift_to_fiber(vorg,x0))
#            print(phif(lg,v))
#            raise
#        bridges[i] = gs
#        log_varphis[i] = log_varphi[-1]
#        Cgvs[i] = np.linalg.norm(phif(lg,v))**2
#        try:
#            v = options['update_v'](v) # update v, e.g. simulate in fiber
#        except KeyError:
#            pass
##     p_T = np.power(2.*np.pi*Tend.eval(),-.5*sigma.shape[0])/np.abs(np.linalg.det(sigma))*np.mean(np.exp(-Cgvs/(2.*Tend.eval()))*np.exp(log_varphis))
#    return -.5*sigma.shape[0]*np.log(2.*np.pi*Tend.eval())-np.log(np.abs(np.linalg.det(sigma)))+np.log(np.mean(np.exp(-Cgvs/(2.*Tend.eval()))*np.exp(log_varphis)))

