import numpy as np

from numba import njit, carray
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
    def __init__ (self, v1, v2):
        self.v1 = v1
        self.v2 = v2
    def __len__ (self):
        return 1
    def make_kernel (self, m, phi, i, t):
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
            m[:] = (1-phi) * v2 * dphi*dphi * (1-phi)
        return dm2

class f12gammadot_model (f12model):
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

class sjoegren_model (model_base):
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
            m[0] = (1-phi[0]) * (2.*v1*phi[0]*dphi[0]+2.*v2*phi[1]*dphi[1]) * (1-phi[0])
            m[1] = (1-phi[1]) * v3 * (phi[0]*dphi[1] + phi[1]*dphi[0]) * (1-phi[1])
        return dm
    def make_dmhat (self, m, f, ehat):
        v1, v2, v3 = self.v1, self.v2, self.v3
        @njit
        def dmhat (m, f, ehat):
            m[0] = ((1-f[0])*2.*v1*f[0]*ehat[0]*(1-f[0]) \
                   +(1-f[1])*v3*f[1]*ehat[1]*(1-f[1]))
            m[1] = ((1-f[0])*2.*v2*f[1]*ehat[0]*(1-f[0]) \
                   +(1-f[1])*v3*f[0]*ehat[1]*(1-f[1]))
        return dmhat
    def make_dm2 (self, m, phi, dphi):
        v1, v2, v3 = self.v1, self.v2, self.v3
        @njit
        def dm2 (m, phi, dphi):
            m[0] = (1-phi[0])*(v1*dphi[0]*dphi[0]+v2*dphi[1]*dphi[1])*(1-phi[0])
            m[1] = (1-phi[1])*v3 * dphi[0]*dphi[1] * (1-phi[1])
        return dm2

