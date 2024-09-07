import numpy as np
import matplotlib.pyplot as plt
import argparse

import mct

parser = argparse.ArgumentParser()
parser.add_argument ('-phi',metavar='phi',help='packing fraction',
                     type=float, nargs='+', default=0.5)
parser.add_argument ('-delta',metavar='delta',help='tagged-particle size',
                     type=float, default=1.0)
args = parser.parse_args()

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

class hssPYtagged (object):
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
        cq1[lowq] = - np.pi/6.*((3*self.Aab-0.5*self.Bab*(1+delta))*(1+delta)**2 - 0.25*self.Dab*delta*(1+delta**2)+(self.eta2**2)*(delta**3)*phi);
        cq2[lowq] = np.pi/240. * ((2.5*self.Aab-0.25*self.Bab*(1+delta))*(1+delta)**4 - (1./24)*self.Dab*delta*(3+10*(delta**2)+3*(delta**4)) + ((delta**3)+(delta**5))*phi*(self.eta2**2)) * q2[lowq]
        cq3[lowq] = -np.pi/26880.*(((7./3)*self.Aab-(1./6)*self.Bab*(1+delta))*(1+delta)**6 -(1./12)*self.Dab*delta*(1+7*(delta**2)+7*(delta**4)+(delta**6)) + (delta**3)*(1./5)*(5+14*(delta**2)+5*(delta**4))*phi * (self.eta2**2)) * q4[lowq]
        return cq1 + cq2 + cq3


def output (d, istart, iend, correlator_array):
    print ("block",d,"\r",end='')


qgrid = np.linspace(0.2,39.8,100)


sqlist = []
cslist = []
for packing_fraction in args.phi:
    sq = hssPY(phi=packing_fraction)
    plt.plot(qgrid, sq.Sq(qgrid)[0])
    cs = hssPYtagged(phi=packing_fraction,delta=args.delta)
    sqlist.append(sq)
    cslist.append(cs)
plt.show()


phi_list = []
phi_s_list = []
msd_list = []
ngp_list = []
for packing_fraction,sq,cs in zip(args.phi,sqlist,cslist):
    # base model "simple liquid"
    model = mct.simple_liquid_model (sq, qgrid)
    phi = mct.correlator (model = model, store = True, blocks=50)
    correlators = mct.CorrelatorStack([phi])

    # tagged-particle model
    model_s = mct.tagged_particle_model (model, cs=cs, D0s=1/args.delta)
    phi_s = mct.correlator (model = model_s, base=phi, store = True)
    correlators.append(phi_s)

    # q=0 limit of tagged particle to determine the MSD
    model_s0 = mct.tagged_particle_q0 (model_s)
    msd = mct.mean_squared_displacement (model = model_s0, base=phi_s, store=True)
    correlators.append(msd)

    # non-Gaussian parameter
    model_ngp = mct.tagged_particle_ngp (model_s0)
    ngp = mct.non_gaussian_parameter (model = model_ngp, base=msd, store=True)
    correlators.append(ngp)

    if False:
        f = mct.nonergodicity_parameter (model = model)
        f.solve()
        fs = mct.nonergodicity_parameter (model = model_s)
        fs.solve()
        plt.plot(qgrid, f.f[0])
        plt.plot(qgrid, fs.f[0])
        plt.show()
    
        ev = mct.eigenvalue (f)
        ev.solve()
        print ("eigenvalue",ev.eval,ev.eval2)
        print ("lambda",ev.lam)
        plt.plot(qgrid,ev.e*(1-f.f[0])**2)
        plt.plot(qgrid,ev.ehat)
        plt.show()


    print("packing fraction",packing_fraction)
    correlators.solve_all(callback=output)
    print("")

    # save:
    #phi.save("/tmp/phi.h5")

    # restore:
    #phi2 = mct.correlator.load('/tmp/phi.h5')
    #correlators2 = mct.CorrelatorStack([phi2])
    #model_s2 = mct.tagged_particle_model (phi2.model, cs=sq)
    #phi_s2 = mct.correlator (model = model_s2, base=phi2, store = True)
    #correlators2.append(phi_s2)
    #correlators2.solve_all(callback=output)
    #print("")

    phi_list.append(phi)
    phi_s_list.append(phi_s)
    msd_list.append(msd)
    ngp_list.append(ngp)


qval = 7.4
qi = np.nonzero(np.isclose(qgrid,qval))[0][0]
for packing_fraction,phi,phi_s in zip(args.phi,phi_list,phi_s_list):
    plt.plot(phi.t, phi.phi[:,qi], label='phi(t), phi={}'.format(packing_fraction))
    plt.plot(phi_s.t, phi_s.phi[:,qi], label='phi_s(t), phi={}'.format(packing_fraction))
    plt.plot(phi.t, np.exp(-qval**2*phi.t),color='black',linestyle='dashed')
plt.xscale('log')
plt.legend()
plt.show()

for packing_fraction,phi,phi_s in zip(args.phi,phi_list,phi_s_list):
    for corr in [phi,phi_s]:
        tau_indices = corr.phi.shape[0] - np.sum(corr.phi<=0.1,axis=0)
        tau_indices[tau_indices>=corr.phi.shape[0]] = -1
        tau = corr.t[tau_indices]
        plt.plot(model.q,tau)
plt.yscale('log')
plt.show()

for packing_fraction,msd in zip(args.phi,msd_list):
    plt.plot(msd.t, msd.phi[:,0])
    plt.plot(msd.t, 6*msd.t/args.delta, color='black', linestyle='dashed')
plt.xscale('log')
plt.yscale('log')
plt.show()

for packing_fraction,ngp in zip(args.phi,ngp_list):
    plt.plot(ngp.t[ngp.t>1e-5], (ngp.phi[:,0][ngp.t>1e-5]/ngp.phi[:,1][ngp.t>1e-5]**2 - 1), label='NGP phi={}'.format(packing_fraction))
plt.xscale('log')
plt.legend()
plt.show()
