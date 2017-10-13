
from src.setup import *
from src.params import *
from src.manifold import *
from src.metric import *
from src.utils import *

from src.Stochastic_Development import *

from src.Regression.params import *
from src.Regression.Processes import *

# Predictions for each observation:
rs = np.random.RandomState(1234)
rng = theano.tensor.shared_randomstreams.RandomStreams(rs.randint(999999))
def Predi(xT,dwt,x0,para):
    
    drift = para[1:(m+1)]
    W = para[(m+1):(m*m+m+1)].reshape((m,m))
    y0 = para[(m*m+m+1):(m*m+m+d+1)]
    ui = para[(m*m+m+d+1):]
    
    q = T.concatenate([y0.reshape((1,d)),ui.reshape((1,d*m))], axis = 1).flatten()
    
    #dwt = rng.normal(size = (n_steps,m), avg = 0, std = T.sqrt(dt))
    Xt = BrownBridge(dwt.flatten(),x0,xT)
    Zt = RegProc(Xt.flatten(),x0,W,drift)
    Ut = stoc_dev(q,Zt[1],drift)
    
    return Ut[-1,0:d]

def Pred(yP,xT,para,y):
    
    tau = para[0]
    #dwt = rng.normal(size = (n_samples,n_steps,m), avg = 0, std = T.sqrt(dt))
    #(cout, updates) = theano.scan(fn=Predi,
    #                              sequences=[xT,dwt],
    #                              non_sequences=[x0,para],
    #                              n_steps=n_samples)
    
    M1 = 1./2*T.mean(y - yP, axis = 0)**2
    M2 = 1./2*T.mean(xT*(y -yP), axis = 0)**2
    M3 = 1./2*(1./(y.shape[0]-2)*T.sum((y - yP)**2, axis = 0) - tau**2)**2
    
    return 1./M1.shape[0]*T.sum(M1) + 1./M2.shape[0]*T.sum(M2) + 1./M3.shape[0]*T.sum(M3)

dwt = T.matrix()
para = T.vector()
y = T.matrix()
x0 = T.vector()
xT = T.matrix()
yP = T.matrix()
Predf = theano.function([yP,xT,para,y], Pred(yP,xT,para,y))

#gradP = lambda para,y,x0,xT: T.grad(Pred(para,y,x0,xT),para)
#gradPf = theano.function([para,y,x0,xT], gradP(para,y,x0,xT))

#n_sim = theano.shared(10)
##pred0 = theano.shared(np.zeros(n_sim.eval()))
#def fopt(para,y,x0,xT,pred0,b):
#    
#    if b == 0:
#        n_sim1 = 10 #dwt = rng.normal(size = (n_sim,n_samples,n_steps,m))
#    else:
#        n_sim1 = n_sim
#    
#    dwt = rng.normal(size = (n_sim1,n_samples,n_steps,m), avg = 0, std = T.sqrt(dt))
#    
#    (cout, updates) = theano.scan(fn=Pred,
#                                 sequences=[dwt],
#                                 non_sequences=[xT,x0,para,y],
#                                 n_steps=n_sim1)
#    
#    if b == 0:
#        pred0 = T.concatenate([cout,pred0[n_sim-10:,:]])
#    else:
#        pred0 = cout
#    
#    return 1./pred0.shape[0]*T.sum(pred0), pred0
#
#b = T.scalar()
#pred0 = T.vector()
#foptf = theano.function([para,y,x0,xT,pred0,b], fopt(para,y,x0,xT,pred0,b),
#                       on_unused_input='ignore')

#from theano.ifelse import ifelse
#def fopt(y,xT,x0,para,yP,rand,b):
#    
#    def foo1(rand,yP,xT,x0,para):
#        dwt = rng.normal(size = (rand.shape[0],n_steps,m), avg = 0, std = T.sqrt(dt))
#        (cout, updates) = theano.scan(fn=Predi,
#                                      sequences=[xT[rand],dwt],
#                                      non_sequences=[x0,para],
#                                      n_steps=rand.shape[0])
#        yP = T.set_subtensor(yP[rand,:],cout)
#        return yP
#    
#    def foo2(yP,xT,x0,para):
#        dwt = rng.normal(size = (n_samples,n_steps,m), avg = 0, std = T.sqrt(dt))
#        (cout, updates) = theano.scan(fn=Predi,
#                                      sequences=[xT,dwt],
#                                      non#_sequences=[x0,para],
#                                      n_steps=n_samples)
#        yP = cout
#        return yP
#    
#    yP = ifelse(T.eq(b,1), foo1(rand,yP,xT,x0,para), foo2(yP,xT,x0,para))
#    
#    return Pred(yP,xT,para,y), yP
#rand = T.ivector()
#b = T.scalar()
#yP = T.matrix()
#foptf = theano.function([y,xT,x0,para,yP,rand,b], fopt(y,xT,x0,para,yP,rand,b),
#                       on_unused_input='ignore')

#def Optimization(init,tol,maxIter,gamma,y,x0,xT):
#    
#    paraN = np.zeros((maxIter,init.shape[0]))
#    para1 = init
#    opt1 = Predf(para1,y,x0,xT)
#
#    for i in range(0,maxIter):
#        
#        paraN[i,:] = para1
#        print("i = ", i)
#
#        start1 = time.time()
#        grad = gradPf(para1,y,x0,xT)
#        diff1 = time.time() - start1
#        print("time = ", diff1)
#
#        paraN[i,1] = para1[1] - gamma*grad[1]/np.linalg.norm(grad[1])
#
#        print("para = ", np.round(paraN[i,:],4))
#        opt2 = Predf(paraN[i,:],y,x0,xT)
#        print("funcval = ", opt2, ": Diff parameters = ", np.dot(paraN[i,:] - para1,paraN[i,:] - para1))
#     
#        if opt2 <= opt1:
#            if np.dot(paraN[i,:] - para1,paraN[i,:] - para1) < tol:
#                print("i break = ", i, ": Converged")
#                break
#
#            para1 = paraN[i,:]
#            opt1 = opt2
#        if opt2 > opt1:
#            gamma = gamma/2
#            print("OBS!! Gamma halved = ", gamma)
#        
#    if i == (maxIter-1):
#        print("Maxiterations reached")
#        
#    return np.round(paraN,4)
