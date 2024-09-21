import numpy as np
from numba import njit

from .solver import correlator

@njit
def _solve_block_g (istart, iend, h, Bq, Wq, g, dG, lambda_, sigma, delta, maxiter, accuracy, calc_moments, alpha, dx, *kernel_args):

    for i in range(istart,iend):

        ibar = i//2
        C = - g[i-1] * dG[1]
        for k in range(2,ibar+1):
            C += (g[i-k+1] - g[i-k]) * dG[k]
        C *= 2
        if (i-ibar > ibar):
            C += (g[i-ibar] - g[i-ibar-1]) * dG[k]
        C += g[i-ibar] * g[ibar]
        C += delta * h*i - sigma

        if alpha==0:
            gg = dG[1]/lambda_
            g[i] = gg - np.sqrt(gg*gg + C/lambda_)
        else:
            C -= alpha/dx**2 * stencil(g[i])
            A = dG[1] + alpha/dx**2
            iterations = 0
            converged = False
            newg = g[i-1]
            while (not converged and iterations < maxiter):
                g[i] = newg
                #kernel (m[i], phi[i], i, h*i, *kernel_args)
                newg = B*g[i] - C
                iterations += 1
                if np.isclose (newg, g[i],
                               rtol=accuracy, atol=accuracy).all():
                    converged = True
                    g[i] = newg
        if calc_moments:
            dG[i] = 0.5 * (g[i-1] + g[i])

class beta_scaling_function (correlator):
    """Solver for the beta-scaling function of MCT.

    Parameters
    ----------
    lam : float, default: 0.735
        Exponent parameter, must be in the range (0.5,1).
    sigma : float, default: 0.0
        Distance parameter to the glass transition.
    delta : float, default: 0.0
        Hopping parameter, must be non-negative.
    t0 : float, default: 1.0
        Time scale for the solutions, must be positive.

    Notes
    -----
    The other parameters take the same meaning as in
    :py:class:`mctspy.correlator`. Note that this solver here
    does not allow to specify a model: the beta-scaling equation
    is model-independent.
    """
    def __init__ (self, blocksize=256, h=1e-9, blocks=60, Tend=0.0,
                  maxinit=50, maxiter=10000, accuracy=1e-9, store=False,
                  lam=0.735, sigma=0., delta=0., t0=1.):
        correlator.__init__ (self, blocksize=blocksize, h=h, blocks=blocks,
            Tend=Tend, maxinit=maxinit, maxiter=maxiter, accuracy=accuracy,
            store=store, model=model_base(), base=None)
        self.lambda_ = lam
        self.sigma = sigma
        self.delta = delta
        self.t0 = t0

    def initial_values (self, imax=50):
        iend = imax
        if (iend >= self.halfblocksize): iend = self.halfblocksize-1
        self.model.set_base(self.phi_)
        a, _ = exponents(self.lambda_)
        self.phi_[0] = 0.0
        for i in range(1,iend):
            t = i*self.h0
            self.phi_[i] = np.power(t/self.t0,-a)
        self.dPhi_[1] = np.power(self.h0/self.t0,-a)/(1-a)
        for i in range(2,iend+1):
            t = i*self.h0
            self.dPhi_[i] = np.power(self.h/self.t0,-a)/(1-a) \
                            * (np.power(1./i,a-1) - np.power(i-1.,1.-a))
        self.iend = iend

    def solve_block (self, istart, iend):
        _solve_block_g (istart, iend, self.h, self.model.Bq()/self.h, self.model.Wq(), self.phi_, self.dPhi_, self.lambda_, self.sigma, self.delta, self.maxiter, self.accuracy, (istart<self.blocksize//2), 0, 0, *self.model.kernel_extra_args())

    def type (self):
        return 'g'

