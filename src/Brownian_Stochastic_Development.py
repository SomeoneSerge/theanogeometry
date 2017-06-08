#from src.group import *
from src.utils import *
from src.manifold import *
from src.metric import *
from src.Stochastic_Development import *

#################################################################
# Simulation of Brownian motion based on Stochastic development #
#################################################################

def SD_brownian(q,dim=2,dWt=None):
    
    if dWt is None:
        dWt = np.random.normal(0, np.sqrt(dt.eval()), (n_steps.get_value(),dim))

    xs = stoc_devf(q,dWt,np.zeros(d.eval()))
    
    return xs

