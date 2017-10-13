
from src.setup import *
from src.params import *
from src.manifold import *
from src.metric import *
from src.utils import *
from src.Stochastic_Development import *

from src.Regression.params import *
from src.Regression.Processes import *

y = T.matrix()
yi = T.vector()
dwt = T.matrix() # 1. dim: number of observations, 2. dim: number of steps times m..
dwtv = T.vector()
Btv = T.vector()
x0 = T.vector()
xTi = T.vector()
xT = T.matrix()
para = T.vector()

def hi(dwti,yi,xTi,x0,para):
    
    Bti = BrownBridge(dwti,x0,xTi)
    Btim = Bti.reshape((n_steps+1,m))
    
    t = T.arange(0, n_steps+1,1)*dt
    Tend = t[-1]
    
    tau = para[0]
    drift = para[1:(m+1)]
    W = para[(m+1):(m*m+m+1)].reshape((m,m))
    y0 = para[(m*m+m+1):(m*m+m+d+1)]
    Xa = para[(m*m+m+d+1):]
    
    q = T.concatenate([y0.reshape((1,d)),Xa.reshape((1,d*rank))], axis = 1).flatten()

    (cout0, updates0) = theano.scan(fn=lambda x: T.extra_ops.diff(x, n=1, axis=0),
                                    sequences=[Btim.T],
                                    n_steps=m) 
    # Had to do this in order for the diff function to be differentiable..
    dBt = cout0.T
    
    Zt = RegProc(Bti,x0,W,drift)
    # Stochastic Development:
    Ut = stoc_dev(q,Zt[1],drift)
    
    logpyi = T.log((2*pi*tau**2)**(-d/2)) - 1./(2*tau**2)*T.dot(yi-Ut[-1,0:d],yi-Ut[-1,0:d])
    
    dBt2 = T.diag(T.tensordot(dBt,dBt, axes = [0,0]))
    # Density for B_t:
    logBt = m*T.log((2*pi*dt.eval())**(-n_steps.get_value()/2)) - T.sum(1./(2*dt.eval())*dBt2)
    # Density for B_T = x_T given B_t:
    logBTg = T.log((2*pi*dt.eval())**(-m/2)) - 1./(2*dt.eval())*T.dot(xTi - Btim[-1,:],xTi - Btim[-1,:])
    # Density for B_T = x_T:
    logBT = T.log((2*pi*Tend)**(-m/2)) - 1./(2*Tend)*T.dot(xTi - x0,xTi - x0)
    
    logBtg = logBt + logBTg - logBT
    
    return logBtg + logpyi

hif = theano.function([dwtv,yi,xTi,x0,para], hi(dwtv,yi,xTi,x0,para), on_unused_input='ignore')

def h(dwtv,y,xT,x0,para):
    
    dwt = dwtv.reshape((n_samples,n_steps*m))
    (cout, updates) = theano.scan(fn=hi,
                                  sequences=[dwt,y,xT],
                                  outputs_info = None,
                                  non_sequences=[x0,para],
                                  n_steps=n_samples)
    
    return -1./n_samples*T.sum(cout)

hf = theano.function([dwtv,y,xT,x0,para], h(dwtv,y,xT,x0,para),
                     on_unused_input='ignore')
