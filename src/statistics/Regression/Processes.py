
from src.statistics.Regression.params import *

from src.setup import *
from src.params import *
#from src.manifold import *
#from src.metric import *
from src.utils import *

def initialize(M):
    dwt = T.matrix()
    dwtv = T.vector()
    Btv = T.vector()
    t = T.scalar()
    Tend = T.scalar()
    x0 = T.vector()
    xT = T.vector()
    Wk = T.matrix() # Covariance matrix..
    drift = T.vector()
    
    def BridgeSde_f(dwt,t,x,Tend,end):
        
        return  x - (x-end)/(Tend-t)*dt.eval() + dwt 
    
    M.BridgeSde_f = BridgeSde_f
    
    M.BridgeSde_ff = theano.function([dwtv,t,x0,Tend,xT], M.BridgeSde_f(dwtv,t,x0,Tend,xT))
    
    # Brownian bridge:
    def BrownBridge(dwtv,x,end):
        
        t = T.arange(0,n_steps.get_value()+1,1)*dt.eval()
        Tend = t[-1]
        
        print(mx.get_value())
        dwt = dwtv.reshape((n_steps.get_value(),mx.get_value()))
        (cout, updates) = theano.scan(fn=M.BridgeSde_f,
                                  sequences=[dwt,t],
                                  outputs_info=[x],
                                  non_sequences=[Tend,end],
                                  n_steps=n_steps.get_value())
        
        Bt = T.concatenate([x.reshape((1,mx.get_value())),cout], axis = 0)
        
        return Bt
    
    M.BrownBridge = BrownBridge
    M.BrownBridgef = theano.function([dwtv,x0,xT], M.BrownBridge(dwtv,x0,xT))
    
    def RegProc(Btv,x,Wk,drift=np.zeros(mx.get_value())):
       
        Bt = Btv.reshape((n_steps.get_value()+1,mx.get_value())) 
        #Bt = BrownBridge(dwtv,x,end)
        (cout0, updates0) = theano.scan(fn=lambda x: T.extra_ops.diff(x, n=1, axis=0),
                                        sequences=[Bt.T],
                                        n_steps=mx.get_value())
        dBt = cout0.T
        
        WdBt = T.tensordot(Wk,dBt, axes = [0,1]).dimshuffle(1,0) 
        
        Zt = T.cumsum(T.concatenate([x.reshape((1,mx.get_value())),WdBt + drift*dt.eval()], axis = 0), axis = 0)
        
        return Zt, WdBt
    
    M.RegProc = RegProc
    M.RegProcf = theano.function([Btv,x0,Wk,drift], M.RegProc(Btv,x0,Wk,drift))
