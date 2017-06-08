from src.setup import *

##########################################################################
# this file contains various object definitions, and standard parameters #
##########################################################################

########### group ###############
invariance = 'left' # chosen metric invariance, right/left

########### manifold ###############
rank = theano.shared(2)

# timestepping
Tend = T.constant(1.)
n_steps = theano.shared(100)
dt = Tend/n_steps

# Integrator variables:
default_method = 'euler'


