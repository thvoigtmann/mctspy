import numpy as np

class hssPY (object):
    """Hard-sphere structure factor, Percus-Yevick approximation.

    Parameters
    ----------
    phi : float
        Packing fraction of the hard-sphere system.
    """
    def __init__ (self, phi):
        etacmp = (1.-phi)**4
        self.alpha = (1.+2*phi)*(1.+2*phi) / etacmp
        self.beta = - (1.+0.5*phi)*(1.+0.5*phi) * 6.*phi/etacmp
        self.phi = phi
        self.lowq = 0.05
    def density (self):
        return self.phi*6/np.pi
    def cq (self, q):
        """Return the direct-correlation function (DCF).

        Parameters
        ----------
        q : array_like
            Grid of wave numbers where the DCF should be evaluated.

        Returns
        -------
        cq : array_like
            DCF evaluated on the given grid.
        """
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
               (cosq/q2[highq] - 4*sinq/q3[highq] - 12*cosq/q4[highq] \
                - 24*((1-cosq) - q[highq]*sinq)/q6)
        cq3[highq] = 8.*np.pi*self.beta * \
               (0.5*cosq/q2[highq] - sinq/q3[highq] + (1-cosq)/q4[highq])
        # for low q, calculate expansion in q
        cq1[lowq] = -np.pi*self.alpha/3. *(4.+self.phi) - np.pi*self.beta
        cq2[lowq] = (np.pi*self.alpha * (2./15. + self.phi/24.) \
                     + np.pi*self.beta/9) * q2[lowq]
        cq3[lowq] = -(np.pi * self.alpha * (1./210. + self.phi/600.) \
                     + np.pi*self.beta/240.)*q4[lowq]
        return cq1 + cq2 + cq3
    def Sq (self, q):
        """Return the structure factor and DCF.

        Parameters
        ----------
        q : array_like
            Grid of wave numbers where S(q) and DCF should be evaluated.

        Returns
        -------
        sq : array_like
            S(q) evaluated on the given grid.
        cq : array_like
            c(q) evaluated on the given grid.
        """
        cq_ = self.cq(q)
        sq_ = 1.0 / (1.0 - self.phi*6./np.pi * cq_)
        # old code used explicit low-q expansion, but since we use
        # a low-q expansion of cq, this doesn't really help much
        #lowq = q<self.lowq
        #q2 = q[lowq]**2
        #q4 = q[lowq]**4
        #eta = self.phi
        #etasq = eta**2
        #eta2 = (1+2.*eta)**2
        #etacmp = (1.-eta)**4
        #sq_[lowq] = etacmp/eta2 * (1 + eta/eta2 * \
        #                           ((16.-11.*eta+4*etasq)/20.*q2 + \
        #                           (-20.+386*eta-627*etasq+494*eta*etasq \
        #                          -etasq*etasq*(173.-21*eta))/(700*eta2)*q4))
        return sq_, cq_
    def dcq_dq (self, q):
        """Return derivative of the DCF.

        Parameters
        ----------
        q : array_like
            Grid of wave numbers where the DCF should be evaluated.

        Returns
        -------
        cq : array_like
            Derivative of the DCF evaluated on the given grid.
        """
        q2 = q*q;
        q3 = q2*q;
        highq = q>=self.lowq
        lowq = q<self.lowq
        q4 = q2[highq]**2
        q6 = q4*q2[highq]
        cosq, sinq = np.cos(q[highq]), np.sin(q[highq]);
        dcq = np.zeros_like(q)
        dcq[highq] = -4*np.pi*self.alpha * \
            (sinq/q2[highq] + 3.*(cosq/q2[highq] - sinq/q3[highq])/q[highq]) \
            - 2*self.phi * np.pi*self.alpha * \
                (sinq/q2[highq] + 6*(cosq/q2[highq] - 4*sinq/q3[highq] \
                - 12*cosq/q4 - 24*((1-cosq) - q[highq]*sinq)/q6)/q[highq]) \
            - 8*np.pi*self.beta * (0.5*sinq/q2[highq] + 2*(cosq/q2[highq] \
                                   + 2*((1-cosq) - q[highq]*sinq)/q4)/q[highq])
        dcq[lowq] = (np.pi*self.alpha * (4./15. + self.phi/12.) \
                     + 2.*np.pi*self.beta/9)*q[lowq] \
                  - (np.pi*self.alpha * (2./105. + self.phi/150.) \
                     + np.pi*self.beta/60.)*q3[lowq];
        return dcq


class hssPYtagged (object):
    """Direct correlation function of tagged hard-sphere particle (PY).

    This implement the specific case of the Percus-Yevick approximation
    of the hard-sphere mixture, where the given particle is a tracer
    in a system of unit-sized spheres.

    Parameters
    ----------
    phi : float
        Packing fraction of the host system.
    delta : float
        Size ratio of tracer to host particles.
    """
    def __init__ (self, phi, delta):
        etacmp = (1.-phi)
        self.eta2 = (1.+2*phi)/etacmp**2
        self.Aab = 0.5*(1-phi+delta*(1+2*phi))/etacmp**2
        self.Bab = (etacmp**2-3*phi*delta*(1+2*phi))/etacmp**3
        self.Dab = 6*phi*(2+phi+delta*(1+2*phi))/etacmp**3
        self.a2 = 6*phi/np.pi/etacmp**2 *\
                  (1.+6*phi/etacmp+9*phi**2/etacmp**2)
        self.phi = phi
        self.delta = delta
        self.lowq = 0.05
    def cq (self, q):
        """Return the direct-correlation function (DCF).

        Parameters
        ----------
        q : array_like
            Grid of wave numbers where the DCF should be evaluated.

        Returns
        -------
        cq : array_like
            DCF evaluated on the given grid.
        """
        highq = q>=self.lowq
        lowq = q<self.lowq
        phi = self.phi
        delta = self.delta
        q2 = q*q
        q3 = q2[highq]*q[highq]
        q4 = q2*q2
        q6 = q4[highq]*q2[highq]
        c1 = np.cos(0.5*q[highq])
        s1 = np.sin(0.5*q[highq])
        cd = np.cos(0.5*q[highq]*delta)
        sd = np.sin(0.5*q[highq]*delta)
        cq1 = np.zeros_like(q)
        cq2 = np.zeros_like(q)
        cq3 = np.zeros_like(q)
        cq1[highq] = 4*np.pi * self.Aab * (c1*cd - s1*sd)/q2[highq]
        cq2[highq] = -4*np.pi * self.Bab * (cd*s1 + c1*sd)/q3 \
                     -4*np.pi * self.Dab * s1 * sd / q4[highq]
        cq3[highq] = -4*np.pi*np.pi * self.a2 * \
                     (q[highq]*c1-2*s1)*(q[highq]*delta*cd-2*sd) / q6
        # low q expansion
        cq1[lowq] = - np.pi/6.*((3*self.Aab-0.5*self.Bab*(1+delta)) \
                    *(1+delta)**2 - 0.25*self.Dab*delta*(1+delta**2) \
                    +(self.eta2**2)*(delta**3)*phi)
        cq2[lowq] = np.pi/240. * ((2.5*self.Aab-0.25*self.Bab*(1+delta))\
                    *(1+delta)**4 - (1./24)*self.Dab*delta*(3+10*(delta**2)\
                    +3*(delta**4)) + ((delta**3)+(delta**5))*phi\
                    *(self.eta2**2)) * q2[lowq]
        cq3[lowq] = -np.pi/26880.*(((7./3)*self.Aab-(1./6)*self.Bab*(1+delta))\
                    *(1+delta)**6 -(1./12)*self.Dab*delta*(1+7*(delta**2)\
                    +7*(delta**4)+(delta**6)) + (delta**3)*(1./5)*(5+14\
                    *(delta**2)+5*(delta**4))*phi * (self.eta2**2)) * q4[lowq]
        return cq1 + cq2 + cq3
