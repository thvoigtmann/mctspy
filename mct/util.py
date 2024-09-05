import numpy as np
from .solver import _decimize

class CorrelatorStack(list):
    def solve_all (self, callback=lambda d,i1,i2,corr:None):
        if not len(self): return
        blocksize = self[0].blocksize
        halfblocksize = self[0].halfblocksize
        blocks = self[0].blocks
        for _phi_ in self:
            _phi_.solve_first()
        callback (0, 0, halfblocksize, self)
        for d in range(blocks):
            for _phi_ in self:
                _phi_.solve_next (d)
            callback (d, halfblocksize, blocksize, self)
            for _phi_ in self:
                _phi_.decimize ()

