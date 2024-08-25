import numpy as np

from numba import njit

from .__util__ import void


@njit(cache=True)
def _decimize (phi, m, dPhi, dM, blocksize):
    halfblocksize = blocksize//2
    imid = halfblocksize//2
    for i in range(1,imid):
        di = i+i
        dPhi[i] = 0.5 * (dPhi[di-1] + dPhi[di])
        dM[i] = 0.5 * (dM[di-1] + dM[di])
    for i in range(imid,halfblocksize):
        di = i+i
        dPhi[i] = 0.25 * (phi[di-2] + 2*phi[di-1] + phi[di])
        dM[i] = 0.25 * (m[di-2] + 2*m[di-1] + m[di])
    dPhi[halfblocksize] = phi[blocksize-1]
    dM[halfblocksize] = m[blocksize-1]
    for i in range(halfblocksize):
        di = i+i
        phi[i] = phi[di]
        m[i] = m[di]
@njit(cache=True)
def _solve_block (istart, iend, h, nutmp, phi, m, dPhi, dM, kernel, maxiter, accuracy):
    A = dM[1] + 1.0 + 1.5*nutmp
    B = (-dPhi[1] + 1.0) / A

    for i in range(istart,iend):
        ibar = i//2
        C = -(m[i-1]*dPhi[1] + phi[i-1]*dM[1])
        for k in range(2,ibar+1):
            C += (m[i-k+1] - m[i-k]) * dPhi[k]
            C += (phi[i-k+1] - phi[i-k]) * dM[k]
        if (i-ibar > ibar):
            C += (phi[i-ibar] - phi[i-ibar-1]) * dM[k]
        C += m[i-ibar] * phi[ibar]
        C += (-2.0*phi[i-1] + 0.5*phi[i-2]) * nutmp
        C = C/A

        iterations = 0
        converged = False
        newphi = phi[i-1]
        while (not converged and iterations < maxiter):
            phi[i] = newphi
            m[i] = kernel (phi[i], i, h*i)
            newphi = B*m[i] - C
            iterations += 1
            if np.isclose (newphi, phi[i],
                           rtol=accuracy, atol=accuracy).all():
                converged = True
                phi[i] = newphi


class dummy_model (object):
    def __len__ (self):
        return 1
    def m (self, phi=0, i=0, t=0):
        return 0.0

class correlator (object):
    def __init__ (self, blocksize=256, h=1e-9, blocks=60,
                  maxiter=10000, accuracy=1e-9, store=False,
                  nu=1.0, kernel = dummy_model, base = None):
        if base is None:
            self.h0 = h
            self.blocks = blocks
            self.halfblocksize = int(blocksize/4)*2
            self.blocksize = self.halfblocksize * 2
        else:
            self.h0 = base.h0
            self.blocks = base.blocks
            self.halfblocksize = base.halfblocksize
            self.blocksize = base.blocksize
            self.base = base
        self.h = self.h0
        self.mdimen = len(kernel)
        self.phi_ = np.zeros((self.blocksize,self.mdimen))
        self.m_ = np.zeros((self.blocksize,self.mdimen))
        self.dPhi_ = np.zeros((self.halfblocksize+1,self.mdimen))
        self.dM_ = np.zeros((self.halfblocksize+1,self.mdimen))
        self.store = store
        if store:
            self.t = np.zeros(self.halfblocksize*(self.blocks+1))
            self.phi = np.zeros((self.halfblocksize*(self.blocks+1),self.mdimen))
            self.m = np.zeros((self.halfblocksize*(self.blocks+1),self.mdimen))

        self.maxiter = maxiter
        self.accuracy = accuracy

        self.nu = nu
        self.kernel = kernel.m
        #def kernel_factory(phi, i, t):
        #    @njit
        #    def ker(phi, i, t):
        #        return kernel.make_m(phi, i, t)
        #    return ker
        #self.jit_kernel = kernel_factory(self.phi[0],0,0.)
        self.jit_kernel = kernel.make_kernel(self.phi_[0],0,0.)

    def phi_addr (self):
        return void(self.phi_)

    def initial_values (self, imax=50):
        iend = imax
        if (iend > self.halfblocksize): iend = self.halfblocksize
        for i in range(iend):
             t = i*self.h0
             self.phi_[i] = np.ones(self.mdimen) - t/self.nu
             #self.m[i] = self.kernel (self.phi[i], i, t)
             self.m_[i] = self.jit_kernel (self.phi_[i], i, t)
        for i in range(1,iend):
             self.dPhi_[i] = 0.5 * (self.phi_[i-1] + self.phi_[i])
             self.dM_[i] = 0.5 * (self.m_[i-1] + self.m_[i])
        self.dPhi_[iend] = self.phi_[iend-1]
        self.dM_[iend] = self.m_[iend-1]
        self.iend = iend

    def decimize (self):
        _decimize (self.phi_, self.m_, self.dPhi_, self.dM_, self.blocksize)
        self.h = self.h * 2.0

    def solve_block (self, istart, iend):
        _solve_block (istart, iend, self.h, self.nu/self.h, self.phi_, self.m_, self.dPhi_, self.dM_, self.jit_kernel, self.maxiter, self.accuracy)

    def solve_all (self, correlators, callback=lambda d,i1,i2,corr:None):
        blocksize = self.blocksize
        halfblocksize = self.halfblocksize
        blocks = self.blocks
        for _phi_ in correlators:
            _phi_.initial_values ()
            _phi_.solve_block (_phi_.iend, halfblocksize)
            if _phi_.store:
                _phi_.t[:halfblocksize] = _phi_.h * np.arange(halfblocksize)
                _phi_.phi[:halfblocksize,:] = _phi_.phi_[:halfblocksize,:]
                _phi_.m[:halfblocksize,:] = _phi_.m_[:halfblocksize,:]
        callback (0, 0, halfblocksize, correlators)
        for d in range(blocks):
            for _phi_ in correlators:
                _phi_.solve_block (_phi_.halfblocksize, _phi_.blocksize)
                if _phi_.store:
                    _phi_.t[d*halfblocksize+halfblocksize:d*halfblocksize+blocksize] = _phi_.h * np.arange(halfblocksize,blocksize)
                    _phi_.phi[d*halfblocksize+halfblocksize:d*halfblocksize+blocksize,:] = _phi_.phi_[halfblocksize:blocksize,:]
                    _phi_.m[d*halfblocksize+halfblocksize:d*halfblocksize+blocksize,:] = _phi_.m_[halfblocksize:blocksize,:]
            callback (d, halfblocksize, blocksize, correlators)
            for _phi_ in correlators:
                _phi_.decimize ()

@njit(cache=True)
def _msd_solve_block (istart, iend, h, nutmp, phi, m, dPhi, dM, kernel, maxiter, accuracy):
    A = dM[1] + 1.5*nutmp

    for i in range(istart,iend):
        ibar = i//2
        C = (m[i]-m[i-1])*dPhi[1] - phi[i-1]*dM[1]
        for k in range(2,ibar+1):
            C += (m[i-k+1] - m[i-k]) * dPhi[k]
            C += (phi[i-k+1] - phi[i-k]) * dM[k]
        if (i-ibar > ibar):
            C += (phi[i-k+1] - phi[i-k]) * dM[k]
        C += m[i-ibar] * phi[ibar]
        C += (-2.0*phi[i-1] + 0.5*phi[i-2]) * nutmp
        C += -6.0
        C = C/A

        m[i] = kernel (None, i, h*i)
        phi[i] = - C


class mean_squared_displacement (correlator):

    def initial_values (self, imax=50):
        iend = imax
        if (iend > self.halfblocksize): iend = self.halfblocksize
        for i in range(iend):
            t = i*self.h0
            self.phi_[i] = np.ones(self.mdimen)*6.*t/self.nu
            self.m_[i] = self.jit_kernel (None, i, t)
        for i in range(1,iend):
            self.dPhi_[i] = 0.5 * (self.phi_[i-1] + self.phi_[i])
            self.dM_[i] = 0.5 * (self.m_[i-1] + self.m_[i])
        self.dPhi_[iend] = self.phi_[iend-1]
        self.dM_[iend] = self.m_[iend-1]
        self.iend = iend

    def solve_block (self, istart, iend):
        _msd_solve_block (istart, iend, self.h, self.nu/self.h, self.phi_, self.m_, self.dPhi_, self.dM_, self.jit_kernel, self.maxiter, self.accuracy)



@njit
def _fsolve (f, m, kernel, M, accuracy, maxiter):
    iterations = 0
    converged = False
    newf = np.ones(M)
    while (not converged and iterations < maxiter):
        iterations+=1
        f = newf
        m = kernel (f)
        newf = m / (1.0+m)
        if np.isclose (newf, f, rtol=accuracy, atol=0.0).all():
            converged = True
            f = newf
        if not iterations%100: print("iter",iterations,f[0])
    return f,m

class nonergodicity_parameter (object):
    def __init__ (self, model, accuracy=1e-12, maxiter=1000000):
        self.model = model
        self.accuracy = accuracy
        self.maxiter = maxiter
        self.f = np.zeros(len(model))
        self.m = np.zeros(len(model))
        self.jit_kernel = model.make_kernel(self.f)

    def solve (self):
        self.f,self.m = _fsolve(self.f, self.m, self.jit_kernel, len(self.model), self.accuracy, self.maxiter)

