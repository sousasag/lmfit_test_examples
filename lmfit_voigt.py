#!/usr/bin/env python
from numpy import loadtxt
from lmfit import fit_report
from lmfit.models import GaussianModel, VoigtModel
import matplotlib.pyplot as plt

data = loadtxt('test_peak.dat')
x = data[:, 0]
y = data[:, 1]

mod = VoigtModel()
params = mod.make_params()

for par in params.values():
    print(par)

out1 = mod.fit(y, params, x=x)

print( 'With Voigt: ')
print( fit_report(out1.params, min_correl=0.25))
print( 'Chi-square = %.3f, Reduced Chi-square = %.3f' % (out1.chisqr, out1.redchi))

plt.plot(x, y, 'ko')
plt.plot(x, out1.best_fit, 'b-')

# make gamma variable
params['gamma'].value = 0.7111
params['gamma'].vary = True
params['gamma'].expr = None

#init = mod.eval(pars, x=x)
out2 = mod.fit(y, params, x=x)

print( 'With Voigt, varying gamma: ')
print( fit_report(out2.params, min_correl=0.25))
print( 'Chi-square = %.3f, Reduced Chi-square = %.3f' % (out2.chisqr,out2.redchi))
plt.plot(x, out2.best_fit, 'g-')
plt.show()
