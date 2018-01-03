# # This file is part of Theano Geometry
#
# Copyright (C) 2017, Stefan Sommer (sommer@di.ku.dk)
# https://bitbucket.org/stefansommer/theanogemetry
#
# Theano Geometry is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Theano Geometry is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Theano Geometry. If not, see <http://www.gnu.org/licenses/>.
#

#from src.group import *
from src.setup import *
from src.utils import *
#from src.manifold import *
#from src.metric import *
#from src.stochastic import Stochastic_Development import *

def initialize(M):
    """ Numerical Brownian motion based on Stochastic development """
    d = M.dim

    def SD_brownian(q,c=1,dim=d.eval(),dWt=None):
    
        if dWt is None:
            dWt = c*np.random.normal(0, np.sqrt(dt.eval()), (n_steps.get_value(),dim))

        xs = M.stoc_devf(q,dWt,np.zeros(dim))
    
        return xs
    M.SD_brownian = SD_brownian
