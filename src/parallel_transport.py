from src.setup import *
from src.utils import *
from src.manifold import *
from src.metric import *

def ode_partrans(gamma,dgamma,t,x):
    
    dpt = - T.tensordot(T.tensordot(dgamma, Gamma_gM(gamma),axes = [0,1]),
                        x, axes = [1,0])

    return dpt

pt = lambda v,gamma,dgamma: integrate(ode_partrans,v,gamma,dgamma)
partrans = lambda v,gamma,dgamma: pt(v,gamma,dgamma)[1]
v = T.vector()
gamma = T.matrix()
dgamma = T.matrix()
partransf = theano.function([v,gamma,dgamma], partrans(v,gamma,dgamma))

if manifold == "S2":
    def plotpar(gamma,vt,v0):
        gammas2 = np.zeros((gamma.shape[0],3))
        for i in range(gamma.shape[0]):
            gammas2[i,:] = Ff(gamma[i,:])

        ax = plt.gca(projection='3d')
        plotx(gamma)
        JFgammai = JFf(gamma[0,:])
        ui = np.dot(JFgammai,v0)#/np.linalg.norm(v0))
        ui = ui#/np.linalg.norm(ui)
        ax.quiver(gammas2[0,0],gammas2[0,1],gammas2[0,2],ui[0],ui[1],ui[2],
                  pivot='tail',
                  arrow_length_ratio = 0.15, length=0.5,
                  color='black')
        sg = np.array([20,40,60,80,99])
        for i in range(5):
            JFgammai = JFf(gamma[sg[i],:])
            ui = np.dot(JFgammai,vt[sg[i],:])#/np.linalg.norm(vt[sg[i],:]))
            ui = ui#/np.linalg.norm(ui)
            ax.quiver(gammas2[sg[i],0],gammas2[sg[i],1],gammas2[sg[i],2],
                      ui[0],ui[1],ui[2],
                      pivot='tail',
                      arrow_length_ratio = 0.15, length=0.5,
                      color='black')
