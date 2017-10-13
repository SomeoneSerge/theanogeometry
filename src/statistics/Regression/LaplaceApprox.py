
from src.setup import *
from src.params import *
from src.manifold import *
from src.metric import *
from src.utils import *

from src.Regression.Processes import *
from src.Regression.params import *
from src.Regression.hfunction import *

y = T.matrix()
yi = T.vector()
dwt = T.matrix() # 1. dim: number of observations, 2. dim: number of steps times m..
dwtv = T.vector()
Btv = T.vector()
x0 = T.vector()
xTi = T.vector()
xT = T.matrix()
para = T.vector()

# Inverse log hessian of hi:
def hesshi(dwtv,yi,xTi,x0,para):
    
    return -nlinalg.LogAbsDet()(-1./n_samples*T.hessian(hi(dwtv,yi,xTi,x0,para),dwtv))
hesshif = theano.function([dwtv,yi,xTi,x0,para], hesshi(dwtv,yi,xTi,x0,para))

# Log determinant of inverse hessian:
def logdethesshf(dwtv,y,xT,x0,para):
    
    dwt = dwtv.reshape((n_samples.get_value(),n_steps.get_value()*m.get_value()))
    logdethess = np.zeros(n_samples.get_value())
    for i in range(n_samples.get_value()):
        logdethess[i] = hesshif(dwt[i,:],y[i,:],xT[i,:],x0,para)
    
    return np.sum(logdethess)

# Laplace Approximation of log-likelihood:
def loglikef(para,x0,y,xT,dwtOp):
    
    # Orthonormal frame:
    Frame = para[(m.get_value()*m.get_value()+m.get_value()+d.eval()+1):].reshape((d.eval(),rank))
    q1 = para[(m.get_value()*m.get_value()+m.get_value()+1):(m.get_value()*m.get_value()+m.get_value()+d.eval()+1)]
    OrthFrame = GramSchmidt(Frame,q1)

    paraO = np.concatenate([para[:(m.get_value()*m.get_value()+m.get_value()+d.eval()+1)].reshape((1,(m.get_value()*m.get_value()+m.get_value()+d.eval()+1))),
                           (OrthFrame.flatten()).reshape((1,d.eval()*rank))], axis = 1).flatten()

    
    dens = hf(dwtOp,y,xT,x0,paraO)
    logdethessm = logdethesshf(dwtOp,y,xT,x0,paraO)
    
    return n_samples.get_value()*dens - 1./2*logdethessm - n_samples.get_value()*n_steps.get_value()*m.get_value()/2*np.log(2*pi/n_samples.get_value())

## Finite differencing of log-likelihood:
from scipy.optimize import approx_fprime

# Optimal paths:
def gh(dwtv,yi,xTi,x0,para):
	
    return scipy.optimize.approx_fprime(dwtv, hif,0.002,yi,xTi,x0,para) 

def gradDec(yi,xTi,x0,para,alpha,maxIter):
    
    points = np.zeros((n_steps.get_value()+1,m.get_value()))
    for i in range(m.get_value()):
        points[:,i] = xTi[i]*np.arange(0,n_steps.get_value()+1.,1)/n_steps.get_value()
    dwtv = np.diff(points, axis = 0).flatten()
    
    Lf = hif(dwtv,yi,xTi,x0,para)
    print("Start likelihood = ", Lf)
    dwtvf = dwtv
    for i in range(maxIter):
        
        dwtvN = dwtvf - alpha*gh(dwtvf,yi,xTi,x0,para)
        LN = hif(dwtvN,yi,xTi,x0,para)
        
        if np.sum(np.abs(dwtvN - dwtvf)) < 10**(-2):
            print("sucessfully terminated")
            break
        if np.sum(np.abs(dwtvN - dwtvf)) >= 10**(-2):
            dwtvf = dwtvN
            Lf = LN
        
        print("like = ", Lf)
    if i == (maxIter-1):
        print("maximal iterations reached")
        
    return dwtvN

def multprocdwti(para,x0,pars):
    
    (yi,xTi) = pars  
    res = gradDec(yi,xTi,x0,para,0.01,10)
    
    return res

def multprocdwt(para,x0,y,xT,n_pool):
    
    p = Pool(processes = n_pool)
    sol = p.imap(partial(multprocdwti,para,x0),\
                         mpu.inputArgs(y,xT),chunksize = n_samples.get_value()/n_pool)
    res = list(sol)
    p.terminate()

    return np.array(zip(res))[:,0,:]

# Lalpace Estimation:
def LapApprox(init,tol,maxIter,gamma,x0,y,xT,dwtOp):

    paraN = np.zeros((maxIter,init.shape[0]))
    para1 = init
    like1 = fop(para1)

    for i in range(0,maxIter):

        print("i = ", i)

        start1 = time.time()
        gradW = scipy.optimize.approx_fprime(para1,loglikef,0.01,x0,y,xT,dwtOp)
        diff1 = time.time() - start1
        print("time = ", diff1)

        paraN[i,:] = para1 - gamma*gradW/np.linalg.norm(gradW)

        print("para = ", np.round(paraN[i,:],4))
        like2 = loglikef(paraN[i,:],x0,y,xT,dwtOp))
        print("funcval = ", like2, ": Diff parameters = ", np.dot(paraN[i,:] - para1,paraN[i,:] - para1))
     
        if like2 <= like1:
            if np.dot(paraN[i,:] - para1,paraN[i,:] - para1) < tol:
                print("i break = ", i, ": Converged")
                break

            para1 = paraN[i,:]
            like1 = like2
        if like2 > like1:
            gamma = gamma/2
            print("OBS!! Gamma halved = ", gamma)
        
    if i == (maxIter-1):
        print("Maxiterations reached")
        
    return np.round(paraN,4)
