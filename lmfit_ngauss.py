#!/usr/bin/python
## Implementation of a function to fit n gaussians (Normalized absorption lines - 1 (to put continuum at 0))
## using lmfit (https://lmfit.github.io/lmfit-py/)

##imports:

from lmfit.models import GaussianModel

## My functions:


def lmfit_ngauss(x,y, params):
  """
  INPUT:
  x - is the wavelength array
  y - is the normalized flux
  params - is a list/array of initial guess values for the parameters
  		   (this controls the number of gaussians to be fitted
  		   	number of gaussians: len(params)/3 - 3 parameters per Gaussian)
  OUTPUT:
  mod - the lmfit model object used for the fit
  out - the lmfit fit object that contains all the results of the fit
  init- array with the initial guess model (usefull to see the initial guess when plotting)
  """
  mods = []
  prefixes = []
  for i in range(0, len(params), 3):
    pref = "g%02i_" % (i/3)
    gauss_i = GaussianModel(prefix=pref)

    if i == 0:
      pars = gauss_i.guess(y, x=x)
    else:
      pars.update(gauss_i.make_params())

    A = params[i]
    l_cen = params[i+1]
    sigma = params[i+2]

    pars[pref+'amplitude'].set(A)
    pars[pref+'center'].set(l_cen)
    pars[pref+'sigma'].set(sigma)

    mods.append(gauss_i)
    prefixes.append(pref)

  mod = mods[0]

  if len(mods) > 1:
    for m in mods[1:]:
      mod += m
  init = mod.eval(pars, x=x)
  out = mod.fit(y, pars, x=x)

  return mod, out, init



def lmfit_ngauss_constrains(x,y, params, constrains):
  """
  INPUT:
  x - is the wavelength array
  y - is the normalized flux
  params - is a list/array of initial guess values for the parameters
  		   (this controls the number of gaussians to be fitted
  		   	number of gaussians: len(params)/3 - 3 parameters per Gaussian)
  contrains - the limits of the constrains for the fit of the parameters
  OUTPUT:
  mod - the lmfit model object used for the fit
  out - the lmfit fit object that contains all the results of the fit
  init- array with the initial guess model (usefull to see the initial guess when plotting)
  """

  mods = []
  prefixes = []
  for i in range(0, len(params), 3):
    pref = "g%02i_" % (i/3)
    gauss_i = GaussianModel(prefix=pref)

    if i == 0:
      pars = gauss_i.guess(y, x=x)
    else:
      pars.update(gauss_i.make_params())
    A = params[i]
    limA = constrains[i]
    l_cen = params[i+1]
    limL = constrains[i+1]
    sigma = params[i+2]
    limS = constrains[i+2]

    pars[pref+'amplitude'].set(A, min=limA[0], max=limA[1])
    pars[pref+'center'].set(l_cen, min=limL[0], max=limL[1])
    pars[pref+'sigma'].set(sigma, min=limS[0], max=limS[1])

    mods.append(gauss_i)
    prefixes.append(pref)

  mod = mods[0]

  if len(mods) > 1:
    for m in mods[1:]:
      mod += m
  init = mod.eval(pars, x=x)
  out = mod.fit(y, pars, x=x)

  return mod, out, init



### Main program:
def main():
  print "Hello"
  

if __name__ == "__main__":
    main()

