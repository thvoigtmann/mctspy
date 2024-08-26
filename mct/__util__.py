import numpy as np
import numba as nb

# https://stackoverflow.com/questions/61509903/how-to-pass-array-pointer-to-numba-function
# we need this to keep references in jit-compiled models that
# reference underlying data of a base model
@nb.extending.intrinsic
def address_as_void_pointer(typingctx, src):
    """ returns a void pointer from a given memory address """
    from numba.core import types, cgutils
    sig = types.voidptr(src)

    def codegen(cgctx, builder, sig, args):
        return builder.inttoptr(args[0], cgutils.voidptr_t)
    return sig, codegen

# take 3-tuple of addres, shape, dtype
@nb.njit
def nparray(ast):
  return nb.carray(address_as_void_pointer(ast[0]),ast[1],ast[2])

def void(nparray):
  return nparray.ctypes.data, nparray.shape, nparray.dtype


class model_base (object):
    def __len__ (self):
        return 1
    def get_kernel (self, phi, i=0, t=0.0):
        if not '__m__' in dir(self):
            self.__m__ = self.make_kernel(phi,i,t)
        return self.__m__
    def get_dm (self, phi, dphi):
        if not '__dm__' in dir(self):
            self.__dm__ = self.make_dm(phi,dphi)
        return self.__dm__
    def get_dmhat (self, f, ehat):
        if not '__dmhat__' in dir(self):
            self.__dmhat__ = self.make_dmhat(f,ehat)
        return self.__dmhat__
    def get_dm2 (self, phi, dphi):
        if not '__dm2__' in dir(self):
            self.__dm2__ = self.make_dm2(phi,dphi)
        return self.__dm2__

    def make_kernel (self, phi, i, t):
        M = len(self)
        @nb.njit
        def dummy(phi, i, t):
            return np.zeros(M)
        return dummy
    def make_dm (self, phi, dphi):
        M = len(self)
        @nb.njit
        def dummy(phi, dphi):
            return np.zeros(M)
        return dummy
    def make_dmhat(self, f, ehat):
        M = len(self)
        @nb.njit
        def dummy(f, ehat):
            return np.zeros(M)
        return dummy
    def make_dm2 (self, phi, dphi):
        M = len(self)
        @nb.njit
        def dummy(phi, dphi):
            return np.zeros(M)
        return dummy
