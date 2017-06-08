
from src.setup import *
from src.params import *

################################
# Setup for Euclidean space    #
################################

manifold = 'euclidean'

d = T.constant(2)

gM = lambda q: T.eye(d)
