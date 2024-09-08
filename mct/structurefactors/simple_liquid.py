import numpy as np

class hssPY (object):
    def __init__ (self, phi):
        etacmp = (1.-phi)**4
        self.alpha = (1.+2*phi)*(1.+2*phi) / etacmp
        self.beta = - (1.+0.5*phi)*(1.+0.5*phi) * 6.*phi/etacmp
        self.phi = phi
        self.lowq = 0.05
    def density (self):
        return self.phi*6/np.pi
    def cq (self, q):
        highq = q>=self.lowq
        lowq = q<self.lowq
        q2 = q*q
        q3 = q2*q
        q4 = q2*q2
        q6 = q4[highq]*q2[highq]
        cosq = np.cos(q[highq])
        sinq = np.sin(q[highq])
        cq1 = np.zeros_like(q)
        cq2 = np.zeros_like(q)
        cq3 = np.zeros_like(q)
        cq1[highq] = 4.*np.pi*self.alpha * (cosq/q2[highq] - sinq/q3[highq])
        cq2[highq] = 2.*np.pi*self.alpha*self.phi * \
               (cosq/q2[highq] - 4*sinq/q3[highq] - 12*cosq/q4[highq] - 24*((1-cosq) - q[highq]*sinq)/q6)
        cq3[highq] = 8.*np.pi*self.beta * (0.5*cosq/q2[highq] - sinq/q3[highq] + (1-cosq)/q4[highq])
        # for low q, calculate expansion in q
        cq1[lowq] = -np.pi*self.alpha/3. *(4.+self.phi) - np.pi*self.beta
        cq2[lowq] = (np.pi*self.alpha * (2./15. + self.phi/24.) + np.pi*self.beta/9) * q2[lowq]
        cq3[lowq] = -(np.pi * self.alpha * (1./210. + self.phi/600.) + np.pi*self.beta/240.)*q4[lowq]
        return cq1 + cq2 + cq3
    def Sq (self, q):
        cq_ = self.cq(q)
        # bigger than low-q
        return 1.0 / (1.0 - self.phi*6/np.pi * cq_), cq_
