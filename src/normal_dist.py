from src.utils import *
from src.manifold import *
from src.metric import *
from src.Brownian_Stochastic_Development import *

def normal_dist_sample(n,q,dim=2):
    
    xs = np.zeros((n,q.shape[0]))
    for i in range(n):
        xs[i,:] = SD_brownian(q,dim)[-1,:]
    
    return xs

