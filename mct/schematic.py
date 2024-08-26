import numpy as np

from numba import njit, carray
from .__util__ import nparray, model_base

class schematic (model_base):
    def __init__ (self, func, M=1):
        self.func = func
        self.M = M
    def __len__ (self):
        return self.M
    def make_kernel (self, phi, i, t):
        F = self.func
        @njit
        def ker(phi, i, t):
            return F(phi)
        return ker

@njit
def f12kernel(phi,v1,v2):
    return v1*phi+v2*phi*phi

class f12model (model_base):
    def __init__ (self, v1, v2):
        self.v1 = v1
        self.v2 = v2
    def __len__ (self):
        return 1
    def make_kernel (self, phi, i, t):
        v1 = self.v1
        v2 = self.v2
        @njit
        def m(phi, i, t):
            return  v1*phi + v2*phi*phi
        return m
    def make_dm (self, phi, dphi):
        v1 = self.v1
        v2 = self.v2
        @njit
        def dm(phi, dphi):
            return v1 * dphi + 2*v2 * phi*dphi
        return dm
    def make_dmhat (self, f, ehat):
        v1 = self.v1
        v2 = self.v2
        @njit
        def dmhat(f, ehat):
            return (1-f)*(v1*ehat + 2*v2 * f*ehat)*(1-f)
        return dmhat
    def make_dm2 (self, phi, dphi):
        v2 = self.v2
        @njit
        def dm2(phi, dphi):
            return v2 * dphi*dphi
        return dm2

class f12gammadot_model (f12model):
    def __init__ (self, v1, v2, gammadot=0.0, gammac=0.1):
        f12model.__init__(self, v1, v2)
        self.gammadot = gammadot
        self.gammac = gammac
    def make_kernel (self, phi, i, t):
        v1 = self.v1
        v2 = self.v2
        gammadot = self.gammadot
        gammac = self.gammac
        @njit
        def ker(phi, i, t):
            gt = gammadot*t / gammac
            return (v1*phi + v2*phi*phi) * 1.0/(1.0 + gt*gt)
        return ker

class sjoegren_model (model_base):
    def __init__ (self, vs, base_model):
        self.vs = vs
        self.base = base_model
    def __len__ (self):
        return 1
    def make_kernel (self, phis, i, t):
        vs = self.vs
        base_phi = self.base.phi
        @njit
        def ker(phis, i, t):
            phi = nparray(base_phi)
            return vs * phi[i] * phis
        return ker



class msd_model (model_base):
    def __init__ (self, vs, base_model):
        self.vs = vs
        self.base = base_model
    def __len__ (self):
        return 1
    def make_kernel (self, phis, i, t):
        vs = self.vs
        base_phi = self.base.phi
        @njit
        def ker(phis, i, t):
            phi = nparray(base_phi)
            return vs * phi[i] * phi[i]
        return ker


class bosse_krieger_model (model_base):
    def __init__ (self, v1, v2, v3):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
    def __len__ (self):
        return 2
    def m (self, phi=np.zeros(2), i=0, t=0):
        return np.array([self.v1 * phi[0]*phi[0] + self.v2 * phi[1]*phi[1],
                         self.v3 * phi[0]*phi[1]])
    def dm (self, phi, dphi):
        return np.array([2.*self.v1*phi[0]*dphi[0]+2.*self.v2*phi[1]*dphi[1],
                         self.v3 * (phi[0]*dphi[1] + phi[1]*dphi[0])])
    def dmhat (self, f, ehat):
        return np.array([(1-f[0])*2.*self.v1*f[0]*ehat[0]*(1-f[0])
                         +(1-f[1])*self.v3*f[1]*ehat[1]*(1-f[1]),
                         (1-f[0])*2.*self.v2*f[1]*ehat[0]*(1-f[0])
                         +(1-f[1])*self.v3*f[0]*ehat[1]*(1-f[1])])
    def dm2 (self, phi, dphi):
        return np.array([self.v1*dphi[0]*dphi[0]+self.v2*dphi[1]*dphi[1],
                         self.v3 * dphi[0]*dphi[1]])


