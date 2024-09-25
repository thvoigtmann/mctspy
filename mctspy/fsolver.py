import numpy as np

from numba import njit

from .__util__ import void

@njit
def _fsolve (f, m, W, f0, phi0, kernel, M, accuracy, maxiter, *kernel_args):
    iterations = 0
    converged = False
    newf = f0
    while (not converged and iterations < maxiter):
        iterations+=1
        f[0] = newf
        kernel (m[0], f[0], 0, 0., *kernel_args)
        newf = m[0] / (W+m[0]) * phi0
        if np.isclose (newf, f[0], rtol=accuracy, atol=0.0).all():
            converged = True
            #f[0] = newf
        #if not iterations%100: print("iter",iterations,np.mean(np.abs(newf-f[0])))
    df = np.mean(np.abs(newf-f[0]))
    f[0] = newf
    return df

@njit
def _fsolve_mat (f, m, W, f0, phi0, WS, kernel, M, accuracy, maxiter, *kernel_args):
    iterations = 0
    converged = False
    lowval = False
    newf = f0.copy()
    #newf = phi0.copy()
    print(newf)
    while (not converged and iterations < maxiter):
        iterations+=1
        f[0] = newf
        kernel (m[0], f[0], 0, 0., *kernel_args)
        if 0 and not lowval:
            for q in range(M):
                newf[q] = phi0[q] - np.linalg.inv(W[q] + m[0][q]) @ WS[q]
        else:
            for q in range(M):
                newf[q] = np.linalg.inv(W[q] + m[0][q]) @ m[0][q] @ phi0[q]
        if np.isclose (newf.reshape(-1), f[0].reshape(-1), rtol=accuracy, atol=0.0).all():
            converged = True
            #f[0] = newf
        if np.isclose (newf.reshape(-1), 0.0, rtol=0.0, atol=10*accuracy).all():
            # suspect liquid solution
            lowval = True
    df = np.mean(np.abs(newf-f[0]))
    f[0] = newf
    return df

class nonergodicity_parameter (object):
    """Solver for the nonergodicity parameter of a simple liquid

    This implements a robust iteration scheme to solve for the NEP.

    Parameters
    ----------
    model : object
        MCT model defining the memory kernel for the NEP.
    accuracy : float, default: 1e-12
        Accuracy at which to terminate the iterative solution scheme.
    maxiter : int
        Maximum number of iterations, to ensure termination.
    """
    def __init__ (self, model, accuracy=1e-12, maxiter=1000000):
        self.model = model
        self.accuracy = accuracy
        self.maxiter = maxiter
        self.dim = model.matrix_dimension()
        self.M = len(model)
        self.f_ = np.zeros((1,self.M*self.dim**2),dtype=model.dtype)
        self.m_ = np.zeros((1,self.M*self.dim**2),dtype=model.dtype)
        self.jit_kernel = model.get_kernel()

    def solve (self, callback=None, callback_every=0):
        """Solve for the nonergodicity parameter.

        Result: The object's field `f` will be set to the NEP values,
        and `m` to the corresponding memory kernel.
        """
        self.model.set_base(self.f_)
        if callback is not None and callback_every>0:
            blocks = self.maxiter//callback_every
            block_iter = callback_every
            if self.maxiter > block_iter*blocks:
                print ("dropping",self.maxiter-block_iter*blocks,"iterations")
        else:
            blocks = 1
            block_iter = self.maxiter
        if self.model.scalar():
            f0 = self.model.phi0().copy()
            for b in range(blocks):
                df = _fsolve(self.f_, self.m_, self.model.Wq(), f0, self.model.phi0(), self.jit_kernel, self.M, self.accuracy, block_iter, *self.model.kernel_extra_args())
                if callback is not None:
                    callback(block_iter*(b+1),df)
                if b < blocks-1:
                    f0 = self.f_[0]
            self.f = self.f_[0]
            self.m = self.m_[0]
        else:
            print("START")
            f0 = self.model.phi0()
            print("START",f0)
            for b in range(blocks):
                df = _fsolve_mat(self.f_.reshape(1,-1,self.dim,self.dim), self.m_.reshape(1,-1,self.dim,self.dim), self.model.Wq().reshape(-1,self.dim,self.dim), f0.reshape(-1,self.dim,self.dim), self.model.phi0().reshape(-1,self.dim,self.dim), self.model.WqSq().reshape(-1,self.dim,self.dim), self.jit_kernel, self.M, self.accuracy, block_iter, *self.model.kernel_extra_args())
                print("done",df)
                if callback is not None:
                    callback(block_iter*(b+1),df)
                if b < blocks-1:
                    f0 = self.f_[0]
            self.f = self.f_[0].reshape(-1,self.dim,self.dim)
            self.m = self.m_[0].reshape(-1,self.dim,self.dim)


@njit
def _esolve(e, dm, f, M, maxiter, accuracy):
    iterations = 0
    converged = False
    newe = np.ones(M)
    while (not converged and iterations < maxiter):
        e[:] = newe
        #newe = (1-f)*model.dm(f,self.e)*(1-f)
        dm(newe,f,e)
        norm = np.sqrt(np.dot(newe,newe))
        if norm>1e-10: newe = newe/norm
        if np.isclose (newe, e, rtol=accuracy, atol=0.0).all():
            converged = True
            e[:] = newe
            eval_ = norm
    return eval_

@njit
def _ehatsolve(ehat, dmhat, f, M, maxiter, accuracy):
    iterations = 0
    converged = False
    newehat = np.ones(M)
    while (not converged and iterations < maxiter):
        ehat[:] = newehat
        dmhat(newehat,f,ehat)
        norm = np.sqrt(np.dot(newehat,newehat))
        if norm>1e-10: newehat = newehat/norm
        if np.isclose (newehat,ehat,rtol=accuracy,atol=0.0).all():
            converged = True
            ehat[:] = newehat
            eval2_ = norm
    return eval2_

class eigenvalue (object):
    """Critical eigenvalue solver for a simple liquid.
    """
    def __init__ (self, nep, accuracy=1e-12, maxiter=1000000):
        self.nep = nep
        self.accuracy = accuracy
        self.maxiter = maxiter
        nep.model.set_C (self.nep.f_[0])
        f, m = self.nep.f_[0], self.nep.m_[0]
        self.dm = self.nep.model.get_dm()
        self.dmhat = self.nep.model.get_dmhat()
        self.dm2 = self.nep.model.get_dm2()

    def solve (self):
        """Solve for the critical eigenvectors.

        This calculates the right- and left critical eigenvector, the
        corresponding eigenvalues, and the exponent parameter.
        """
        f = self.nep.f_[0]
        model = self.nep.model
        self.e = np.zeros(len(model),dtype=model.dtype)
        self.ehat = np.zeros(len(model),dtype=model.dtype)
        self.eval = _esolve(self.e, self.dm, f, len(model), self.maxiter, self.accuracy)
        self.eval2 = _ehatsolve(self.ehat, self.dmhat, f, len(model), self.maxiter, self.accuracy)
        if self.eval > 0:
            dq = model.dq()
            nl = np.dot(dq * self.ehat, self.e)
            nr = np.dot(dq * self.ehat, self.e*self.e * (1-f))
            self.e = self.e * nl/nr
            self.ehat = self.ehat * nr/(nl*nl)
            #self.lam = np.dot(self.ehat, (1-f)*self.dm2(f,self.e)*(1-f))
            C = np.zeros(len(model))
            self.dm2(C,f,self.e)
            self.lam = np.dot(dq * self.ehat, C)
        else:
            self.lam = 0.0


