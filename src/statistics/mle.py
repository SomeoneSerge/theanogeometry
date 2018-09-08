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

def iterative_mle(obss,log_p_T,update_thetas,options):
    try:
        log_likelihoods = np.zeros(options['epochs'])
        
        # initial thetas
        thetas = np.array(options['initial'])
        try:
            if options['verbose']:
                print("initial thetas:", thetas)
        except KeyError:
            pass

        chain_sampler = False
        try:
            chain_sampler = options['chain_sampler']
        except KeyError:
            pass
        if chain_sampler:
            chains = np.zeros((obss.shape[0],)+options['chain_size'])
    
        # iterations
        thetass = [np.zeros((options['epochs'],) + theta.shape) for theta in thetas]
        for j, theta in enumerate(thetas):
            thetass[j][0] = theta
        mpu.openPool()
        for i in range(options['epochs']):
            if not chain_sampler:
                sol = mpu.pool.imap(partial(log_p_T, thetas), \
                                    mpu.inputArgs(obss, np.random.randint(1000, size=obss.shape[0])))
            else:
                sol = mpu.pool.imap(partial(log_p_T, thetas), \
                                    mpu.inputArgs(obss, np.random.randint(1000, size=obss.shape[0]), chains))
            res = list(sol)
            log_likelihood = np.mean(mpu.getRes(res, 0), axis=0)
            dthetas = [np.mean(mpu.getRes(res, j + 1), axis=0) for j in range(len(thetas))]
            if chain_sampler:
                chains = mpu.getRes(res,len(thetas)+1)
            # step, update parameters and varphis
#             print(dthetas)
            thetas = update_thetas(thetas, dthetas)

            # save iterations
            log_likelihoods[i] = log_likelihood  # total log likelihood
            for j, theta in enumerate(thetas):
                thetass[j][i] = theta
            
            try:
                if options['verbose']:
                    print("iteration: ", i, ", log-likelihood: ", log_likelihood)
                    print("thetas:", thetas)
            except KeyError:
                pass
                    
        return (thetas, log_likelihood, log_likelihoods, thetass)
    except:
        mpu.closePool()
        raise
    else:
        mpu.closePool()
