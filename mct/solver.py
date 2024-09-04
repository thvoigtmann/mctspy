import numpy as np

from numba import njit

from .__util__ import model_base


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
def _solve_block (istart, iend, h, Bq, Wq, phi, m, dPhi, dM, kernel, maxiter, accuracy, calc_moments):
    A = dM[1] + Wq + 1.5*Bq / h
    B = (-dPhi[1] + phi[0]) / A

    for i in range(istart,iend):
        ibar = i//2
        C = -(m[i-1]*dPhi[1] + phi[i-1]*dM[1])
        for k in range(2,ibar+1):
            C += (m[i-k+1] - m[i-k]) * dPhi[k]
            C += (phi[i-k+1] - phi[i-k]) * dM[k]
        if (i-ibar > ibar):
            C += (phi[i-ibar] - phi[i-ibar-1]) * dM[k]
        C += m[i-ibar] * phi[ibar]
        C += (-2.0*phi[i-1] + 0.5*phi[i-2]) * Bq/h
        C = C/A

        iterations = 0
        converged = False
        newphi = phi[i-1]
        while (not converged and iterations < maxiter):
            phi[i] = newphi
            kernel (m[i], phi[i], i, h*i)
            newphi = B*m[i] - C
            iterations += 1
            if np.isclose (newphi, phi[i],
                           rtol=accuracy, atol=accuracy).all():
                converged = True
                phi[i] = newphi
                if calc_moments:
                    dPhi[i] = 0.5 * (phi[i-1] + phi[i])
                    dM[i] = 0.5 * (m[i-1] + m[i])


class correlator (object):
    def __init__ (self, blocksize=256, h=1e-9, blocks=60,
                  maxiter=10000, accuracy=1e-9, store=False,
                  nu=1.0, model = model_base, base = None):
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
        self.mdimen = len(model)
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

        self.model = model
        model.set_base(self.phi_)
        self.jit_kernel = model.get_kernel(self.m_[0],self.phi_[0],0,0.0)

    def initial_values (self, imax=50):
        iend = imax
        if (iend > self.halfblocksize): iend = self.halfblocksize
        phi0 = self.model.phi0()
        tauinv = self.model.Wq() * phi0 / self.model.Bq()
        for i in range(iend):
             t = i*self.h0
             self.phi_[i] = phi0 - t*tauinv
             self.jit_kernel (self.m_[i], self.phi_[i], i, t)
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
        _solve_block (istart, iend, self.h, self.model.Bq(), self.model.Wq(), self.phi_, self.m_, self.dPhi_, self.dM_, self.jit_kernel, self.maxiter, self.accuracy,(istart<self.blocksize//2))


@njit(cache=True)
def _msd_solve_block (istart, iend, h, nutmp, phi, m, dPhi, dM, kernel, maxiter, accuracy, calc_moments):
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

        kernel (m[i], None, i, h*i)
        phi[i] = - C
        if calc_moments:
            dPhi[i] = 0.5 * (phi[i-1] + phi[i])
            dM[i] = 0.5 * (m[i-1] + m[i])


class mean_squared_displacement (correlator):

    def initial_values (self, imax=50):
        iend = imax
        if (iend > self.halfblocksize): iend = self.halfblocksize
        for i in range(iend):
            t = i*self.h0
            self.phi_[i] = np.ones(self.mdimen)*6.*t/self.model.Bq()
            self.jit_kernel (self.m_[i], None, i, t)
        for i in range(1,iend):
            self.dPhi_[i] = 0.5 * (self.phi_[i-1] + self.phi_[i])
            self.dM_[i] = 0.5 * (self.m_[i-1] + self.m_[i])
        self.dPhi_[iend] = self.phi_[iend-1]
        self.dM_[iend] = self.m_[iend-1]
        self.iend = iend

    def solve_block (self, istart, iend):
        _msd_solve_block (istart, iend, self.h, self.model.Bq()/self.h, self.phi_, self.m_, self.dPhi_, self.dM_, self.jit_kernel, self.maxiter, self.accuracy, (istart<self.blocksize//2))


