import numpy as np
import matplotlib.pyplot as plt
import argparse

import mct

parser = argparse.ArgumentParser()
parser.add_argument ('-phi',metavar='phi',help='packing fraction',
                     type=float, default=0.5)
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

sq = hssPY(phi=args.phi)

qgrid = np.linspace(0.2,39.8,100)

plt.plot(qgrid, sq.Sq(qgrid)[0])
plt.show()

model = mct.simple_liquid_model (sq, qgrid)
phi = mct.correlator (model = model, store = True, blocks=50)
correlators = mct.CorrelatorStack([phi])

model_s = mct.tagged_particle_model (model, cs=sq)
phi_s = mct.correlator (model = model_s, base=phi, store = True)
correlators.append(phi_s)

model_s0 = mct.tagged_particle_q0 (model_s)
msd = mct.mean_squared_displacement (model = model_s0, base=phi_s, store=True)
correlators.append(msd)

if False:
    f = mct.nonergodicity_parameter (model = model)
    f.solve()
    plt.plot(qgrid, f.f[0])
    plt.show()
    
    ev = mct.eigenvalue (f)
    ev.solve()
    print ("eigenvalue",ev.eval,ev.eval2)
    print ("lambda",ev.lam)
    plt.plot(qgrid,ev.e*(1-f.f[0])**2)
    plt.plot(qgrid,ev.ehat)
    plt.show()

def output (d, istart, iend, correlator_array):
    print ("block",d,"\r",end='')

correlators.solve_all(callback=output)


qval = 7.4
qi = np.nonzero(np.isclose(qgrid,qval))[0][0]
print(phi.phi.shape)
plt.plot(phi.t, phi.phi[:,qi])
plt.plot(phi_s.t, phi_s.phi[:,qi])
plt.plot(phi.t, np.exp(-qval**2*phi.t),color='black',linestyle='dashed')
#plt.plot(phi.t, phi.m[:,qi])
plt.xscale('log')
plt.show()

for corr in [phi,phi_s]:
    tau_indices = corr.phi.shape[0] - np.sum(corr.phi<=0.1,axis=0)
    tau_indices[tau_indices>=corr.phi.shape[0]] = -1
    tau = corr.t[tau_indices]
    plt.plot(model.q,tau)
plt.yscale('log')
plt.show()

plt.plot(msd.t, msd.phi[:,0])
plt.plot(msd.t, 6*msd.t, color='black', linestyle='dashed')
plt.xscale('log')
plt.yscale('log')
plt.show()