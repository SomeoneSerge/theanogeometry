
from src.setup import *
from src.params import *
from src.utils import *

#from src.stochastic_development import *

from src.statistics.Regression.params import *
from src.statistics.Regression.Processes import *

def initialize(M):
    dwt = T.matrix()
    para = T.vector()
    y = T.matrix()
    x0 = T.vector()
    xT = T.matrix()
    yP = T.matrix()

    # Predictions for each observation:
    rs = np.random.RandomState(1234)
    rng = theano.tensor.shared_randomstreams.RandomStreams(rs.randint(999999))
    def Predi(xT,dwt,x0,para):
        
        drift = para[1:(mx.get_value()+1)]
        W = para[(mx.get_value()+1):(mx.get_value()*mx.get_value()+mx.get_value()+1)].reshape((mx.get_value(),mx.get_value()))
        y0 = para[(mx.get_value()*mx.get_value()+mx.get_value()+1):(mx.get_value()*mx.get_value()+mx.get_value()+M.dim.eval()+1)]
        ui = para[(mx.get_value()*mx.get_value()+mx.get_value()+M.dim.eval()+1):]
        
        q = T.concatenate([y0.reshape((1,M.dim.eval())),ui.reshape((1,M.dim.eval()*mx.get_value()))], axis = 1).flatten()
        
        #dwt = rng.normal(size = (n_steps,m), avg = 0, std = T.sqrt(dt))
        Xt = M.BrownBridge(dwt.flatten(),x0,xT)
        Zt = M.RegProc(Xt.flatten(),x0,W,drift)
        Ut = M.stochastic_development(q,Zt[1])[1] # Add drift!
        
        return Ut[-1,0:M.dim.eval()]

    M.Predi = Predi
    
    def Pred(para,xT,x0,y):
    
        tau = para[0]
        dwt = rng.normal(size = (n_samples.get_value(),n_steps.get_value(),mx.get_value()), avg = 0, std = T.sqrt(dt.eval()))
        (cout, updates) = theano.scan(fn=M.Predi,
                                      sequences=[xT,dwt],
                                      non_sequences=[x0,para],
                                      n_steps=n_samples.get_value())
    
        M1 = 1./2*T.mean(y - cout, axis = 0)**2
        M2 = 1./2*(1./n_samples.get_value()*T.dot(xT.T,(y-cout)).flatten())**2
        #M2 = 1./2*T.mean(xT.reshape((n_samples,m))*(y - cout), axis = 0)**2
        M3 = 1./2*(1./(y.shape[0]-2)*T.sum((y - cout)**2, axis = 0) - tau**2)**2
    
        return 1./M1.shape[0]*T.sum(M1) + 1./M2.shape[0]*T.sum(M2) + 1./M3.shape[0]*T.sum(M3)

    #def Pred(yP,xT,para,y):
    #    
    #    tau = para[0]
    #    #dwt = rng.normal(size = (n_samples.get_value(),n_steps.get_value(),mx.get_value()), avg = 0, std = T.sqrt(dt.eval()))
    #    #(cout, updates) = theano.scan(fn=Predi,
    #    #                              sequences=[xT,dwt],
    #    #                              non_sequences=[x0,para],
    #    #                              n_steps=n_samples.get_value())
    #    
    #    M1 = 1./2*T.mean(y - yP, axis = 0)**2
    #    M2 = 1./2*T.mean(xT*(y -yP), axis = 0)**2
    #    M3 = 1./2*(1./(y.shape[0]-2)*T.sum((y - yP)**2, axis = 0) - tau**2)**2
    #    
    #    return 1./M1.shape[0]*T.sum(M1) + 1./M2.shape[0]*T.sum(M2) + 1./M3.shape[0]*T.sum(M3)
    
    M.Pred = Pred
    M.Predf = theano.function([para,xT,x0,y], M.Pred(para,xT,x0,y))
    

    n_sim = theano.shared(10)
    def fopt(para,xT,x0,y):
        
        (cout, updates) = theano.scan(fn=M.Pred,
                                      non_sequences=[para,xT,x0,y],
                                      n_steps=n_sim.get_value())
            
        return 1./cout.shape[0]*T.sum(cout), cout
    
    M.fopt = fopt
    
    #from multiprocess import Pool
    #import src.multiprocess_utils as mpu
    #import itertools
    #from functools import partial
    #def multPred(para,start,y2,end2):
    #    
    #    p = Pool(processes = 2)
    #    #mpu.openPool()
    #    sol = p.imap(partial(Predf,para,xT,x0,y),
    #                         chunksize = n_sim/2)
    #    res = list(sol)
    #    p.terminate()
    #    #mpu.closePool()
    #
    #    return np.array(zip(res))#[:,0,:]
    
    M.foptf = theano.function([para,xT,x0,y], M.fopt(para,xT,x0,y),
                              on_unused_input='ignore')
    
    M.gradPi = lambda para,xT,x0,y: T.grad(M.Pred(para,xT,x0,y),para)
    M.gradPif = theano.function([para,xT,x0,y], M.gradPi(para,xT,x0,y))
    
    def gradP(para,xT,x0,y):
        
        (cout, updates) = theano.scan(fn=M.gradPi,
                                     #sequences=[dwt],
                                     non_sequences=[para,xT,x0,y],
                                     n_steps=n_sim.get_value())
        
        return 1./cout.shape[0]*T.sum(cout, axis = 0)

    M.gradP = gradP
    M.gradPf = theano.function([para,xT,x0,y], M.gradP(para,xT,x0,y))
    
    #def multgrad(para,start,y2,end2):
    #    
    #    p = Pool(processes = 2)
    #    #mpu.openPool()
    #    sol = p.imap(partial(gradPif,para,xT,x0,y),
    #                         chunksize = n_sim/2)
    #    res = list(sol)
    #    p.terminate()
    #    #mpu.closePool()
    #
    #    return np.array(zip(res))#[:,0,:]

