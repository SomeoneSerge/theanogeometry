
from src.Regression.params import *

from src.setup import *
from src.params import *
from src.manifold import *
from src.metric import *
from src.utils import *

dwt = T.matrix()
dwtv = T.vector()
Btv = T.vector()
t = T.scalar()
Tend = T.scalar()
x0 = T.vector()
xT = T.vector()
Wk = T.matrix() # Covariance matrix..

def BridgeSde_f(dwt,t,x,Tend,end):
    
    return  x - (x-end)/(Tend-t)*dt + dwt 

BridgeSde_ff = theano.function([dwtv,t,x0,Tend,xT], BridgeSde_f(dwtv,t,x0,Tend,xT))

# Brownian bridge:
def BrownBridge(dwtv,x,end):
    
    t = T.arange(0,n_steps+1,1)*dt
    Tend = t[-1]
    
    dwt = dwtv.reshape((n_steps,m))
    (cout, updates) = theano.scan(fn=BridgeSde_f,
                              sequences=[dwt,t],
                              outputs_info=[x],
                              non_sequences=[Tend,end],
                              n_steps=n_steps)
    
    Bt = T.concatenate([x.reshape((1,m)),cout], axis = 0)
    
    return Bt

BrownBridgef = theano.function([dwtv,x0,xT], BrownBridge(dwtv,x0,xT))

def RegProc(Btv,x,Wk,drift=np.zeros(m.get_value())):
   
    Bt = Btv.reshape((n_steps+1,m)) 
    #Bt = BrownBridge(dwtv,x,end)
    (cout0, updates0) = theano.scan(fn=lambda x: T.extra_ops.diff(x, n=1, axis=0),
                                    sequences=[Bt.T],
                                    n_steps=m)
    dBt = cout0.T
    
    WdBt = T.tensordot(Wk,dBt, axes = [0,1]).dimshuffle(1,0) 
    
    Zt = T.cumsum(T.concatenate([x.reshape((1,m)),WdBt + drift*dt], axis = 0), axis = 0)
    
    return Zt, WdBt

RegProcf = theano.function([Btv,x0,Wk,drift], RegProc(Btv,x0,Wk,drift))
