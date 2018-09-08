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

from src.setup import *
from src.utils import *

# Hamiltonian MCMC, numeric version
# adapted from http://www.mcmchandbook.net/HandbookChapter5.pdf

def HMC_step_numeric(U, grad_U, noisef, epsilon, L, current_q, extra_params):
    q = np.copy(current_q)
    p = noisef() # independent standard normal variates
    current_p = np.copy(p)

    # Make a half step for momentum at the beginning
    p += -epsilon * grad_U(q,*extra_params) / 2
    # Alternate full steps for position and momentum
    for i in range(L):
        # Make a full step for the position
        q += epsilon * p

        # Make a full step for the momentum, except at end of trajectory
        if i != L:
            p += -epsilon * grad_U(q,*extra_params)

    # Make a half step for momentum at the end.
    p += -epsilon * grad_U(q,*extra_params) / 2

    # Negate momentum at end of trajectory to make the proposal symmetric
    p = -p

    # Evaluate potential and kinetic energies at start and end of trajectory
    current_U = U(current_q,*extra_params)
    current_K = np.sum(current_p**2) / 2
    proposed_U = U(q,*extra_params)
    proposed_K = np.sum(p**2) / 2

    # Accept or reject the state at end of trajectory, returning either
    # the position at the end of the trajectory or the initial position
#     print('current U', current_U)
#     print('current K', current_K)
#     print('proposed U', proposed_U)
#     print('proposed K', proposed_K)
    density = np.exp(current_U-proposed_U+current_K-proposed_K)

    if (np.random.uniform() < density):
#         print('accept', density)
        return (q) # accept
    else:
#         print('reject', density)        
        return (current_q) # reject

# Theano Hamiltonian MCMC
# from http://deeplearning.net/tutorial/hmc.html

def kinetic_energy(vel):
    """Returns the kinetic energy associated with the given velocity
    and mass of 1.

    Parameters
    ----------
    vel: theano matrix
        Symbolic matrix whose rows are velocity vectors.

    Returns
    -------
    return: theano vector
        Vector whose i-th entry is the kinetic entry associated with vel[i].

    """
    return 0.5 * (vel ** 2).sum(axis=1)


def hamiltonian(pos, vel, energy_fn):
    """
    Returns the Hamiltonian (sum of potential and kinetic energy) for the given
    velocity and position.

    Parameters
    ----------
    pos: theano matrix
        Symbolic matrix whose rows are position vectors.
    vel: theano matrix
        Symbolic matrix whose rows are velocity vectors.
    energy_fn: python function
        Python function, operating on symbolic theano variables, used tox
        compute the potential energy at a given position.

    Returns
    -------
    return: theano vector
        Vector whose i-th entry is the Hamiltonian at position pos[i] and
        velocity vel[i].
    """
    # assuming mass is 1
    return energy_fn(pos) + kinetic_energy(vel)


def metropolis_hastings_accept(energy_prev, energy_next, s_rng):
    """
    Performs a Metropolis-Hastings accept-reject move.

    Parameters
    ----------
    energy_prev: theano vector
        Symbolic theano tensor which contains the energy associated with the
        configuration at time-step t.
    energy_next: theano vector
        Symbolic theano tensor which contains the energy associated with the
        proposed configuration at time-step t+1.
    s_rng: theano.tensor.shared_randomstreams.RandomStreams
        Theano shared random stream object used to generate the random number
        used in proposal.

    Returns
    -------
    return: boolean
        True if move is accepted, False otherwise
    """
    ediff = energy_prev - energy_next
    return (T.exp(ediff) - s_rng.uniform(size=energy_prev.shape)) >= 0


def simulate_dynamics(initial_pos, initial_vel, stepsize, HMC_steps, energy_fn):
    """
    Return final (position, velocity) obtained after an `HMC_steps` leapfrog
    updates, using Hamiltonian dynamics.

    Parameters
    ----------
    initial_pos: shared theano matrix
        Initial position at which to start the simulation
    initial_vel: shared theano matrix
        Initial velocity of particles
    stepsize: shared theano scalar
        Scalar value controlling amount by which to move
    energy_fn: python function
        Python function, operating on symbolic theano variables, used to
        compute the potential energy at a given position.

    Returns
    -------
    rval1: theano matrix
        Final positions obtained after simulation
    rval2: theano matrix
        Final velocity obtained after simulation
    """

    def leapfrog(pos, vel, step):
        """
        Inside loop of Scan. Performs one step of leapfrog update, using
        Hamiltonian dynamics.

        Parameters
        ----------
        pos: theano matrix
            in leapfrog update equations, represents pos(t), position at time t
        vel: theano matrix
            in leapfrog update equations, represents vel(t - stepsize/2),
            velocity at time (t - stepsize/2)
        step: theano scalar
            scalar value controlling amount by which to move

        Returns
        -------
        rval1: [theano matrix, theano matrix]
            Symbolic theano matrices for new position pos(t + stepsize), and
            velocity vel(t + stepsize/2)
        rval2: dictionary
            Dictionary of updates for the Scan Op
        """
        # from pos(t) and vel(t-stepsize//2), compute vel(t+stepsize//2)
        dE_dpos = T.grad(energy_fn(pos).sum(), pos)
        new_vel = vel - step * dE_dpos
        # from vel(t+stepsize//2) compute pos(t+stepsize)
        new_pos = pos + step * new_vel
        return [new_pos, new_vel], {}

    # compute velocity at time-step: t + stepsize//2
    initial_energy = energy_fn(initial_pos)
    dE_dpos = T.grad(initial_energy.sum(), initial_pos)
    vel_half_step = initial_vel - 0.5 * stepsize * dE_dpos

    # compute position at time-step: t + stepsize
    pos_full_step = initial_pos + stepsize * vel_half_step

    # perform leapfrog updates: the scan op is used to repeatedly compute
    # vel(t + (m-1/2)*stepsize) and pos(t + m*stepsize) for m in [2,HMC_stes].
    (all_pos, all_vel), scan_updates = theano.scan(
        leapfrog,
        outputs_info=[
            dict(initial=pos_full_step),
            dict(initial=vel_half_step),
        ],
        non_sequences=[stepsize],
        n_steps=HMC_steps - 1)
    final_pos = all_pos[-1]
    final_vel = all_vel[-1]
    # NOTE: Scan always returns an updates dictionary, in case the
    # scanned function draws samples from a RandomStream. These
    # updates must then be used when compiling the Theano function, to
    # avoid drawing the same random numbers each time the function is
    # called. In this case however, we consciously ignore
    # "scan_updates" because we know it is empty.
    assert not scan_updates

    # The last velocity returned by scan is vel(t +
    # (HMC_steps - 1 / 2) * stepsize) We therefore perform one more half-step
    # to return vel(t + HMC_steps * stepsize)
    energy = energy_fn(final_pos)
    final_vel = final_vel - 0.5 * stepsize * T.grad(energy.sum(), final_pos)

    # return new proposal state
    return final_pos, final_vel


# start-snippet-1
def HMC_step(s_rng, positions, energy_fn, stepsize, HMC_steps):
    """
    This function performs one-step of Hybrid Monte-Carlo sampling. We start by
    sampling a random velocity from a univariate Gaussian distribution, perform
    `HMC_steps` leap-frog updates using Hamiltonian dynamics and accept-reject
    using Metropolis-Hastings.

    Parameters
    ----------
    s_rng: theano shared random stream
        Symbolic random number generator used to draw random velocity and
        perform accept-reject move.
    positions: shared theano matrix
        Symbolic matrix whose rows are position vectors.
    energy_fn: python function
        Python function, operating on symbolic theano variables, used to
        compute the potential energy at a given position.
    stepsize:  shared theano scalar
        Shared variable containing the stepsize to use for `HMC_steps` of HMC
        simulation steps.
    HMC_steps: integer
        Number of HMC steps to perform before proposing a new position.

    Returns
    -------
    rval1: boolean
        True if move is accepted, False otherwise
    rval2: theano matrix
        Matrix whose rows contain the proposed "new position"
    """
    # end-snippet-1 start-snippet-2
    # sample random velocity
    initial_vel = s_rng.normal(size=positions.shape)
    # end-snippet-2 start-snippet-3
    # perform simulation of particles subject to Hamiltonian dynamics
    final_pos, final_vel = simulate_dynamics(
        initial_pos=positions,
        initial_vel=initial_vel,
        stepsize=stepsize,
        HMC_steps=HMC_steps.astype('int64'),
        energy_fn=energy_fn
    )
    # end-snippet-3 start-snippet-4
    # accept/reject the proposed move based on the joint distribution
    accept = metropolis_hastings_accept(
        energy_prev=hamiltonian(positions, initial_vel, energy_fn),
        energy_next=hamiltonian(final_pos, final_vel, energy_fn),
        s_rng=s_rng
    )
    # end-snippet-4
    return accept, final_pos

## Example
# HMC_stepsize = T.scalar()
# HMC_steps = T.scalar()
# HMC_stepf = theano.function([dWt,HMC_stepsize,HMC_steps], HMC_step(srng, dWt, lambda w: U(w,x,v), T.sqrt(dt)*HMC_stepsize, HMC_steps))
