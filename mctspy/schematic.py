import numpy as np
import scipy.linalg as la

from numba import njit, carray, objmode
from .__util__ import nparray, model_base

class schematic (model_base):
    def __init__ (self, func, M=1):
        self.func = func
        self.M = M
    def __len__ (self):
        return self.M
    def make_kernel (self, m, phi, i, t):
        F = self.func
        @njit
        def ker(m, phi, i, t):
            m = F(phi)
        return ker

class f12model (model_base):
    r"""The schematic F12 model of mode-coupling theory.

    Declares a single memory kernel of the form
    :math:`m[f] = v_1 f + v_2 f^2`.
    """
    def __init__ (self, v1, v2):
        self.v1 = v1
        self.v2 = v2
    def __len__ (self):
        return 1
    def make_kernel (self, m, phi, i, t):
        """Kernel-factory method.

        Returns njit-able function to evaluate the F12 model."""
        v1 = self.v1
        v2 = self.v2
        @njit
        def ker(m, phi, i, t):
            m[:] = v1*phi + v2*phi*phi
        return ker
    def make_dm (self, m, phi, dphi):
        v1 = self.v1
        v2 = self.v2
        @njit
        def dm(m, phi, dphi):
            m[:] = (1-phi) * (v1 * dphi + 2*v2 * phi*dphi) * (1-phi)
        return dm
    def make_dmhat (self, m, f, ehat):
        v1 = self.v1
        v2 = self.v2
        @njit
        def dmhat(m, f, ehat):
            m[:] = (1-f)*(v1*ehat + 2*v2 * f*ehat)*(1-f)
        return dmhat
    def make_dm2 (self, m, phi, dphi):
        v2 = self.v2
        @njit
        def dm2(m, phi, dphi):
            m[:] = (1-phi)**2 * v2 * dphi*dphi * (1-phi)**2
        return dm2

class f12gammadot_model (f12model):
    r"""Schematic F12-gammadot model.

    This sets a single memory kernel to be
    :math:`m[\phi(t),t] = h(\dot\gamma t)[v_1 \phi(t) + v_2 \phi(t)^2]`.
    The strain-reduction function is defined to be
    :math:`h(\gamma)=1/(1+\gamma^2)`.
    This model was introduced by Fuchs and Cates as a schematic model to
    calculate the dynamics, especially flow curves under strong shear."""
    def __init__ (self, v1, v2, gammadot=0.0, gammac=0.1):
        f12model.__init__(self, v1, v2)
        self.gammadot = gammadot
        self.gammac = gammac
    def make_kernel (self, m, phi, i, t):
        v1 = self.v1
        v2 = self.v2
        gammadot = self.gammadot
        gammac = self.gammac
        @njit
        def ker(m, phi, i, t):
            gt = gammadot*t / gammac
            m[:] = (v1*phi + v2*phi*phi) * 1.0/(1.0 + gt*gt)
        return ker

class f12gammadot_tensorial_model (f12gammadot_model):
    r"""Schematic F12-gammadot model, tensorial version.

    This is the adaptation of the :py:class:`mctspy.f12gammadot_model`
    for tensorial flow fields."""
    def __init__ (self, v1, v2, gammadot=0.0, gammac=0.1,
                  kappa=np.array([[0,1,0],[0,0,0],[0,0,0]],dtype=float),
                  use_hhat=False, nu=1.0):
        f12gammadot_model.__init__(self, v1, v2, gammadot, gammac)
        self.kappa = kappa
        self.use_hhat = use_hhat
        self.nu = nu
    def make_kernel (self, m, phi, i, t):
        v1 = self.v1
        v2 = self.v2
        gammadot = self.gammadot
        gammac = self.gammac
        nu = self.nu
        @njit
        def ker(m, phi, i, t):
            #gt = gammadot*t
            with objmode(ht='float64'):
                #F = la.expm(kappa*gt)
                #B = np.dot(F,F.T)
                #I1 = np.trace(B)
                #I2 = np.trace(la.inv(B))
                ht = self.hhat(t)
            #ht = 1.0 / (1. + (nu*I1 + (1-nu)*I2 - 3)/gammac**2)
            m[:] = (v1*phi + v2*phi*phi) * ht
        return ker
    def hhat (self, t):
        gt = self.gammadot*t
        F = la.expm(self.kappa * gt)
        B = np.dot(F,F.T)
        I1 = np.trace(B)
        I2 = 0.0
        if self.nu != 1.0:
            try:
                I2 = np.trace(la.inv(B))
            except:
                I2 = 0.
        return 1.0 / (1. + (self.nu*I1 + (1-self.nu)*I2 - 3)/self.gammac**2)
    def kernel_prefactor (self, trange):
        if self.use_hhat:
            print ("CALC hhat",trange)
            res = np.array([self.hhat(t) for t in trange])
            print (res)
            print ("DONE")
            return res
        else:
            return np.ones_like(trange)


class sjoegren_model (model_base):
    r"""Sj\ |ouml|\ gren model.

    Defines a memory kernel that couples to a given (usually, F12) base
    model, and sets :math:`m^s[f^s] = v_s f f^c` where :math:`f` is taken
    from the base model."""
    def __init__ (self, vs, base_model):
        self.vs = vs
        self.base = base_model
    def __len__ (self):
        return 1
    def make_kernel (self, ms, phis, i, t):
        vs = self.vs
        base_phi = self.base.phi
        @njit
        def ker(ms, phis, i, t):
            phi = nparray(base_phi)
            ms[:] = vs * phi[i] * phis
        return ker



class msd_model (model_base):
    def __init__ (self, vs, base_model):
        self.vs = vs
        self.base = base_model
    def __len__ (self):
        return 1
    def make_kernel (self, ms, phis, i, t):
        vs = self.vs
        base_phi = self.base.phi
        @njit
        def ker(ms, phis, i, t):
            phi = nparray(base_phi)
            ms[:] = vs * phi[i] * phi[i]
        return ker


class bosse_krieger_model (model_base):
    """Bosse-Krieger model.

    A two-correlator model famous for the discussion of higher-order
    singularities."""
    def __init__ (self, v1, v2, v3):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
    def __len__ (self):
        return 2
    def make_kernel (self, m, phi, i, t):
        v1, v2, v3 = self.v1, self.v2, self.v3
        @njit
        def ker (m, phi, i, t):
            m[0] = v1 * phi[0]*phi[0] + v2 * phi[1]*phi[1]
            m[1] = v3 * phi[0]*phi[1]
        return ker
    def make_dm (self, m, phi, dphi):
        v1, v2, v3 = self.v1, self.v2, self.v3
        @njit
        def dm (m, phi, dphi):
            dp = (1-phi)**2 * dphi
            m[0] = 2.*v1*phi[0]*dp[0] + 2.*v2*phi[1]*dp[1]
            m[1] = v3 * (phi[0]*dp[1] + phi[1]*dp[0])
        return dm
    def make_dmhat (self, m, f, ehat):
        v1, v2, v3 = self.v1, self.v2, self.v3
        @njit
        def dmhat (m, f, ehat):
            m[0] = 2.*v1*f[0]*ehat[0]*(1-f[0])**2 \
                   +v3*f[1]*ehat[1]*(1-f[0])**2
            m[1] = 2.*v2*f[1]*ehat[0]*(1-f[1])**2 \
                   +v3*f[0]*ehat[1]*(1-f[1])**2
        return dmhat
    def make_dm2 (self, m, phi, dphi):
        v1, v2, v3 = self.v1, self.v2, self.v3
        @njit
        def dm2 (m, phi, dphi):
            dp = (1-phi)**2 * dphi
            m[0] = v1*dp[0]*dp[0]+v2*dp[1]*dp[1]
            m[1] = v3 * dp[0]*dp[1]
        return dm2

