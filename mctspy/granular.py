import numpy as np
from numba import njit
import scipy.special

from .standard import simple_liquid_model

class granular_model (simple_liquid_model):
    """Granular MCT model.
    """
    def __init__ (self, Sq, q, vth=1.0, restitution_coeff=1.0):
        self.epsilon = restitution_coeff
        self.vth = vth
        simple_liquid_model.__init__ (self, Sq, q, vth)
        if 'contact_value' in dir(Sq):
            self.contact_value = Sq.contact_value()
        else:
            self.contact_value = 1.0
        self.fixed_motion_type = 'damped_newtonian'
    def Aq (self):
        epsfac = (1+self.epsilon)/2 + (1-self.epsilon)/2*self.sq
        return self.sq/self.vth**2/epsfac
    def Bq (self):
        epsfac = (1+self.epsilon)/2 + (1-self.epsilon)/2*self.sq
        return self.omega_E()*(1+self.epsilon)/2 * \
            (1 - scipy.special.spherical_jn(0,self.q) \
            + 2*scipy.special.spherical_jn(2,self.q)) * \
            self.sq/self.vth**2/epsfac
    def omega_E (self):
        return 4*np.sqrt(np.pi)*self.rho*self.vth * self.contact_value
    def __init_vertices__ (self):
        simple_liquid_model.__init_vertices__(self)
        Aeps = 1. / (1. + (1.-self.epsilon)*self.sq/(1+self.epsilon))
        for n in range(9):
            self.b[n] = self.b[n] * Aeps[:,None]
