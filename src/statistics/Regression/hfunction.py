
from src.setup import *
from src.params import *
#from src.manifold import *
#from src.metric import *
from src.utils import *
from src.stochastics.stochastic_development import *

from src.statistics.Regression.params import *
from src.statistics.Regression.Processes import *

def initialize(M):
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
        
        Bti = M.BrownBridge(dwti,x0,xTi)
        Btim = Bti.reshape((n_steps.get_value()+1,mx.get_value()))
        
        t = T.arange(0, n_steps.get_value()+1,1)*dt.eval()
        Tend = t[-1]
        
        tau = para[0]
        drift = para[1:(mx.get_value()+1)]
        W = para[(mx.get_value()+1):(mx.get_value()*mx.get_value()+mx.get_value()+1)].reshape((mx.get_value(),mx.get_value()))
        y0 = para[(mx.get_value()*mx.get_value()+mx.get_value()+1):(mx.get_value()*mx.get_value()+mx.get_value()+M.dim.eval()+1)]
        Xa = para[(mx.get_value()*mx.get_value()+mx.get_value()+M.dim.eval()+1):]
        
        q = T.concatenate([y0.reshape((1,M.dim.eval())),Xa.reshape((1,M.dim.eval()*mx.get_value()))], axis = 1).flatten()
    
        (cout0, updates0) = theano.scan(fn=lambda x: T.extra_ops.diff(x, n=1, axis=0),
                                        sequences=[Btim.T],
                                        n_steps=mx.get_value()) 
        # Had to do this in order for the diff function to be differentiable..
        dBt = cout0.T
        
        Zt = M.RegProc(Bti,x0,W,drift)
        # Stochastic Development:
        Ut = M.stochastic_development(q,Zt[1])[1] ###### ADD DRIFT!
        
        logpyi = T.log((2*pi*tau**2)**(-M.dim.eval()/2)) - 1./(2*tau**2)*T.dot(yi-Ut[-1,0:M.dim.eval()],yi-Ut[-1,0:M.dim.eval()])
        
        dBt2 = T.diag(T.tensordot(dBt,dBt, axes = [0,0]))
        # Density for B_t:
        logBt = mx.get_value()*T.log((2*pi*dt.eval())**(-n_steps.get_value()/2)) - T.sum(1./(2*dt.eval())*dBt2)
        # Density for B_T = x_T given B_t:
        logBTg = T.log((2*pi*dt.eval())**(-mx.get_value()/2)) - 1./(2*dt.eval())*T.dot(xTi - Btim[-1,:],xTi - Btim[-1,:])
        # Density for B_T = x_T:
        logBT = T.log((2*pi*Tend)**(-mx.get_value()/2)) - 1./(2*Tend)*T.dot(xTi - x0,xTi - x0)
        
        logBtg = logBt + logBTg - logBT
        
        return logBtg + logpyi
    
    M.hi = hi
    M.hif = theano.function([dwtv,yi,xTi,x0,para], M.hi(dwtv,yi,xTi,x0,para), on_unused_input='ignore')
    
    def h(dwtv,y,xT,x0,para):
        
        dwt = dwtv.reshape((n_samples.get_value(),n_steps.get_value()*mx.get_value()))
        (cout, updates) = theano.scan(fn=M.hi,
                                      sequences=[dwt,y,xT],
                                      outputs_info = None,
                                      non_sequences=[x0,para],
                                      n_steps=n_samples.get_value())
        
        return -1./n_samples.get_value()*T.sum(cout)
    
    M.h = h
    M.hf = theano.function([dwtv,y,xT,x0,para], M.h(dwtv,y,xT,x0,para),
                         on_unused_input='ignore')
