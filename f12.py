import numpy as np
import argparse
import matplotlib.pyplot as plt

import mct
from numba import njit


parser = argparse.ArgumentParser()
parser.add_argument ('-v1',metavar='v1',help='vertex v1 of F12 model',
                     type=float, default=0.)
parser.add_argument ('-v2',metavar='v2',help='vertex v2 of F12 model',
                     type=float, default=3.95)
parser.add_argument ('-vs',metavar='vs',help='vertex vs of Sjoegren model',
                     type=float, default=15.0)
parser.add_argument ('-v3',metavar='v3',help='vertex v3 of BK model',
                     type=float, default=45.0)
parser.add_argument ('-gammadot',metavar='gdot',help='shear rate',
                     type=float, default=1e-4)
args = parser.parse_args()

model = mct.f12model (args.v1, args.v2)
#v1,v2 = args.v1,args.v2
#model = mct.schematic (njit(lambda x:v1*x + v2*x*x))
phi = mct.correlator (kernel = model, maxiter=1000000, blocksize=256, accuracy=1e-10, store=True)
correlators = [phi]

#model_s = mct.sjoegren_model(args.vs,phi)
#phi_s = mct.correlator (kernel = model_s, base=phi, store=True)
#correlators.append (phi_s)

#model_msd = mct.schematic.msd_model(args.vs,phi_s)
#msd = mct.mean_squared_displacement (kernel = model_msd, base=phi_s, store=True)
#correlators.append (msd)

#shear_model = mct.f12gammadot_model (args.v1, args.v2, gammadot=args.gammadot)
#phi_gdot = mct.correlator (kernel = shear_model, maxiter=1000000, blocksize=256, accuracy=1e-10, store=True)
#correlators.append(phi_gdot)

def output (d, istart, iend, correlator_array, filter=0):
    first = correlator_array[0]
    for i in range(istart,iend):
        if not (filter and (i%filter)):
            print ("{t:.15f} ".format(t=i*first.h),end='')
            for correlator in correlator_array:
                for q in range(len(correlator.phi[i])):
                    print ("{phi:.15f} ".format(phi=correlator.phi[i][q]),end='')
                for q in range(len(correlator.m[i])):
                    print ("{m:.15f} ".format(m=correlator.m[i][q]),end='')
            print ("#")
def output (d, istart, iend, correlator_array):
    print ("block",d,"\r",end='')
    print ("")


f = mct.nonergodicity_parameter (model)
f.solve()
print(f.f,f.m)

ev = mct.eigenvalue (f)
ev.solve()
print ("# eigenvalue = {:f} (check ehat: {:f})".format(ev.eval,ev.eval2))
print ("# e = {}".format(ev.e))
print ("# ehat = {}".format(ev.ehat))
print ("# lambda = {:f}".format(ev.lam))


phi.solve_all(correlators, callback=output)
print("")

plt.plot(phi.t,phi.phi)
#plt.plot(phi_gdot.t,phi_gdot.phi)
#plt.plot(phi_s.t,phi_s.phi)
plt.xscale('log')
plt.show()

#plt.plot(msd.t,msd.phi)
#plt.xscale('log')
#plt.yscale('log')
#plt.show()
