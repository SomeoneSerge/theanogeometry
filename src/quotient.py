from src.group import *
from src.manifold import *
from src.metric import *

# hit target v at time t=Tend
def get_sde_fiber(sde_f,proj):
    def sde_fiber(dW,t,x,*ys):
        (det,sto,X,*dys_sde) = sde_f(dW,t,x,*ys)
        
        # compute kernel of proj derivative with respect to inv A metric
        rank = d
        Xframe = T.tensordot(invpf(x,eiLA),sigma,(2,0))
        Xframe_inv = T.nlinalg.MatrixPinv()(Xframe.reshape((-1,G_dim)))
        dproj = T.tensordot(T.jacobian(proj(x).flatten(),x),
                            Xframe,
                           ((1,2),(0,1))).reshape((proj(x).shape[0],G_dim))
        (_,_,Vh) = T.nlinalg.svd(dproj,full_matrices=True)
        ns = Vh[rank:].T # null space
        proj_ns = T.tensordot(ns,ns,(1,1))
        
        det = T.tensordot(Xframe,T.tensordot(proj_ns,T.tensordot(Xframe_inv,det.flatten(),(1,0)),(1,0)),(2,0)).reshape(g.shape)
        sto = T.tensordot(Xframe,T.tensordot(proj_ns,T.tensordot(Xframe_inv,sto.flatten(),(1,0)),(1,0)),(2,0)).reshape(g.shape)
        X = T.tensordot(Xframe,T.tensordot(proj_ns,T.tensordot(Xframe_inv,X.flatten(),(1,0)),(1,0)),(2,0)).reshape(X.shape)
        
        return (det,sto,X,*dys_sde)

    return sde_fiber

# find g in fiber above x closests to g0
from scipy.optimize import minimize
def lift_to_fiber(x,x0):
    shoot = lambda hatxi: gVf(hatxi,hatxi)
    try:
        hatxi = minimize(shoot,
                zeroV.eval(),
                method='COBYLA',
                constraints={'type':'ineq','fun':lambda hatxi: np.min((G_injectivity_radius.eval()-np.max(hatxi),
                                                                      1e-8-np.linalg.norm(actf(expf(VtoLAf(hatxi)),x0)-x)**2))},
                ).x
    except NameError: # injectivity radius not defined
        hatxi = minimize(shoot,zeroV.eval()).x
    l0 = expf(VtoLAf(hatxi))
    try: # project to group if to_group function is available
        l0 = to_groupf(l0)
    except NameError:
        pass
    return (l0,hatxi)
