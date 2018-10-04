
from src.statistics.Regression.params import *

from src.setup import *
from src.params import *
#from src.manifold import *
#from src.metric import *
from src.utils import *

def initialize(M):
    
    dwt = T.matrix()
    dwtv = T.vector()
    t = T.scalar()
    Tend = T.scalar()
    x0 = T.vector()
    xT = T.vector()
    Wk = T.matrix() # Covariance matrix..
    drift = T.vector()
    dBt = T.matrix()
        
    def sde_bridge(dwt,t,x,end):
        
        detx = -(x-end)/(1-t)
        sto = dwt
        return  (detx, sto, t, T.constant(0.))
    
    M.sde_bridge = sde_bridge
    M.sde_bridgef = theano.function([dwt,t,x0,xT], M.sde_bridge(dwt,t,x0,xT))
       
    # Brownian bridge:
    def brown_bridge(dwt,x,xT):
    
        # integrator: (returning both Bt and dBt!)
        def euler_c(dW,t,x,*ys):
            (detx, stox, X,*ys) = sde_bridge(dW,t,x,xT)
            return (t+dt,x + dt*detx + stox, dt*detx + stox)
        
        # integration of sde:
        (cout, updates) = theano.scan(fn=euler_c,
                outputs_info=[T.constant(0.),x,T.zeros_like(xT)],
                sequences=[dwt],
                n_steps=n_steps)
        
        dBt = cout[2]
        Bt  = cout[1]
        
        return dBt, Bt
        
    M.brown_bridge = brown_bridge
    M.brown_bridgef = theano.function([dwt,x0,xT], M.brown_bridge(dwt,x0,xT), on_unused_input='ignore')
    
    def reg_euclid_proc(dBt,Wk,drift=np.zeros(mx.get_value())):
            
            WdBt = T.tensordot(Wk,dBt, axes = [0,1]).dimshuffle(1,0) 
            
            dZt = WdBt + drift*dt #T.cumsum(T.concatenate([x.reshape((1,mx.get_value())),WdBt + drift*dt.eval()], axis = 0), axis = 0)
            
            return dZt, WdBt
        
    M.reg_euclid_proc = reg_euclid_proc
    M.reg_euclid_procf = theano.function([dBt,Wk,drift], M.reg_euclid_proc(dBt,Wk,drift))




#dwt = T.matrix()
#    dwtv = T.vector()
#    Btv = T.vector()
#    t = T.scalar()
#    Tend = T.scalar()
#    x0 = T.vector()
#    xT = T.vector()
#    Wk = T.matrix() # Covariance matrix..
#    drift = T.vector()
#    
#    def sde_bridge(dwt,t,x,end):
#    
#    detx = -(x-end)/(1-t)
#    sto = dwt
#    return  (detx, sto, t, T.constant(0))  
#    
#    M.sde_bridge = sde_bridge
#    M.sde_bridgef = theano.function([dwt,t,x0,xT], M.sde_bridge(dwt,t,x0,xT), on_unused_input = 'ignore')
#   
#    # Brownian bridge:
#    def brown_bridge(dwt,x,xT):
#
#        return integrate_sde(M.sde_bridge,integrator_ito,x,dwt,xT)#,Tend,xT) #Bt
#    
#    M.brown_bridge = brown_bridge
#    M.brown_bridgef = theano.function([dwt,x0,xT], M.brown_bridge(dwt,x0,xT), on_unused_input='ignore')
#    
#    def RegProc(Btv,x,Wk,drift=np.zeros(mx.get_value())):
#       
#        Bt = Btv.reshape((n_steps.get_value()+1,mx.get_value())) 
#        #Bt = BrownBridge(dwtv,x,end)
#        (cout0, updates0) = theano.scan(fn=lambda x: T.extra_ops.diff(x, n=1, axis=0),
#                                        sequences=[Bt.T],
#                                        n_steps=mx.get_value())
#        dBt = cout0.T
#        
#        WdBt = T.tensordot(Wk,dBt, axes = [0,1]).dimshuffle(1,0) 
#        
#        Zt = T.cumsum(T.concatenate([x.reshape((1,mx.get_value())),WdBt + drift*dt.eval()], axis = 0), axis = 0)
#        
#        return Zt, WdBt
#    
#    M.RegProc = RegProc
#    M.RegProcf = theano.function([Btv,x0,Wk,drift], M.RegProc(Btv,x0,Wk,drift))
