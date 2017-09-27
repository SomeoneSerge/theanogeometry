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

from src.metric import *

def Frechet_mean(y,Logf,q0=None,options=None):    
    global steps
    steps = []
    steps.append(q0)
    
    if q0 is None:
        q0 = np.random.normal(size=y[0].shape)

    def fopts(x):
        x = x.reshape(q0.shape)
        N = y.shape[0]
        sol = mpu.pool.imap(lambda pars: (Logf(x,y[pars[0]],np.zeros(x.shape))[0],),mpu.inputArgs(range(N)))
        res = list(sol)
        Logs = mpu.getRes(res,0)     

        res = (1./N)*np.sum(np.square(Logs))        
        grad = -(2./N)*np.sum(Logs,0)

        return (res,grad)

    def save_step(k):
        global steps
        steps.append(k)
   
    try:
        mpu.openPool()
        res = minimize(fopts, q0, method='BFGS', jac=True, options=options, callback=save_step)
    except:
        mpu.closePool()
        raise
    else:
        mpu.closePool()
    
    return(res.x,res.fun,np.array(steps))

