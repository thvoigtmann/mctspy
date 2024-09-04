import numpy as np
from .solver import _decimize

class CorrelatorStack(list):
    def solve_all (self, callback=lambda d,i1,i2,corr:None):
        if not len(self): return
        blocksize = self[0].blocksize
        halfblocksize = self[0].halfblocksize
        blocks = self[0].blocks
        for _phi_ in self:
            #_phi_.initial_values ()
            _phi_.initial_values (imax=halfblocksize)
            _phi_.solve_block (_phi_.iend, halfblocksize)
            if _phi_.store:
                _phi_.t[:halfblocksize] = _phi_.h * np.arange(halfblocksize)
                _phi_.phi[:halfblocksize,:] = _phi_.phi_[:halfblocksize,:]
                _phi_.m[:halfblocksize,:] = _phi_.m_[:halfblocksize,:]
                _phi_.reco = np.zeros_like(_phi_.phi_)
                _phi_.reco_m = np.zeros_like(_phi_.phi_)
                _phi_.dreco = np.zeros_like(_phi_.dPhi_)
                _phi_.dreco_m = np.zeros_like(_phi_.dPhi_)
                _phi_.reco[:halfblocksize,:] = _phi_.phi[:halfblocksize,:]
                _phi_.reco_m[:halfblocksize,:] = _phi_.m[:halfblocksize,:]
                if not _phi_.iend < halfblocksize:
                  _phi_.dreco[1:halfblocksize,:] = (_phi_.reco[:halfblocksize-1,:] + _phi_.reco[1:halfblocksize,:])/2.
                  _phi_.dreco_m[1:halfblocksize,:] = (_phi_.reco_m[:halfblocksize-1,:] + _phi_.reco_m[1:halfblocksize,:])/2.
                  _phi_.dreco[halfblocksize,:] = _phi_.reco[halfblocksize-1,:]
                  _phi_.dreco_m[halfblocksize,:] = _phi_.reco_m[halfblocksize-1,:]
                else:
                  _phi_.dreco[1:halfblocksize,:] = (_phi_.reco[:halfblocksize-1,:] + _phi_.reco[1:halfblocksize,:])/2.
                  _phi_.dreco_m[1:halfblocksize,:] = (_phi_.reco_m[:halfblocksize-1,:] + _phi_.reco_m[1:halfblocksize,:])/2.
                  #_phi_.dreco[1:,:] = (_phi_.reco[:halfblocksize,:] + _phi_.reco[1:halfblocksize+1,:])/2.
                  #_phi_.dreco_m[1:,:] = (_phi_.reco_m[:halfblocksize,:] + _phi_.reco_m[1:halfblocksize+1,:])/2.
                #print(_phi_.dPhi_.reshape(-1))
                #print(_phi_.dreco.reshape(-1))
                #print((_phi_.dreco==_phi_.dPhi_).reshape(-1))
                assert((_phi_.reco==_phi_.phi_).all())
                assert((_phi_.dreco==_phi_.dPhi_).all())
        callback (0, 0, halfblocksize, self)
        for d in range(blocks):
            for _phi_ in self:
                _phi_.solve_block (_phi_.halfblocksize, _phi_.blocksize)
                if _phi_.store:
                    _phi_.t[d*halfblocksize+halfblocksize:d*halfblocksize+blocksize] = _phi_.h * np.arange(halfblocksize,blocksize)
                    _phi_.phi[d*halfblocksize+halfblocksize:d*halfblocksize+blocksize,:] = _phi_.phi_[halfblocksize:blocksize,:]
                    _phi_.m[d*halfblocksize+halfblocksize:d*halfblocksize+blocksize,:] = _phi_.m_[halfblocksize:blocksize,:]
                    _phi_.reco[halfblocksize:,:] = _phi_.phi[d*halfblocksize+halfblocksize:d*halfblocksize+blocksize,:]
                    _phi_.reco_m[halfblocksize:,:] = _phi_.m[d*halfblocksize+halfblocksize:d*halfblocksize+blocksize,:]
                    assert((_phi_.reco==_phi_.phi_).all())
                    assert((_phi_.dreco==_phi_.dPhi_).all())
            callback (d, halfblocksize, blocksize, self)
            for _phi_ in self:
                _phi_.decimize ()
                _decimize(_phi_.reco,_phi_.reco_m,_phi_.dreco,_phi_.dreco_m,blocksize) 
                assert((_phi_.reco==_phi_.phi_).all())

