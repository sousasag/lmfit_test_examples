#!/usr/bin/env python
#<examples/doc_nistgauss.py>
import numpy as np
from lmfit.models import GaussianModel, ExponentialModel
import sys
import matplotlib.pyplot as plt
from numpy import sqrt, pi, exp, linspace, loadtxt
from astropy.io import fits



def gaussian(x, amp, cen, wid):
  "1-d gaussian: gaussian(x, amp, cen, wid)"
  return (amp/(sqrt(2*pi)*wid)) * exp(-(x-cen)**2 /(2*wid**2))


def lmfit_ngauss(x,y, *params):
  params = params[0]
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

  print mod

  init = mod.eval(pars, x=x)
  out = mod.fit(y, pars, x=x)
  return mod, out, init


#-------------------------------------------

def lmfit_ngauss_constrains(x,y, params, constrains):
  #params = params[0]
  #constrains = constrains[0]
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

#-------------------------------------------------
#-------------------------------------------------

def test_2gaussians_with_ngaussians():

  x = np.linspace(0.0, 10.0, num=1000)
  y = gaussian(x, -1, 3, 0.75) + gaussian(x, -0.5, 5, 0.8) + np.random.normal(0, 0.01, x.shape[0])

  params = [-0.9, 2.5, 0.5, -0.4, 5, 0.5]

  mod, out, init = lmfit_ngauss(x,y, params)

  
  plt.plot(x, y)
  plt.plot(x, init, 'k--')

  print(out.fit_report(min_correl=0.5))

  plt.plot(x, out.best_fit, 'r-')
  plt.show()

#-------------------------------------------------
#-------------------------------------------------


def test_2synthlines_with_ngaussians():

  x = np.linspace(5800, 5803, num=1000)
  y = gaussian(x, -0.8, 5801.1, 0.2) + gaussian(x, -0.5, 5802.2, 0.2) + np.random.normal(0, 0.01, x.shape[0])

  params = [-0.9, 5801, 0.1, -0.4, 5802, 0.1]

  mod, out, init = lmfit_ngauss(x,y, params)

  
  plt.plot(x, y)
  plt.plot(x, init, 'k--')

  print(out.fit_report(min_correl=0.5))

  plt.plot(x, out.best_fit, 'r-')
  plt.show()


#-------------------------------------------------
#-------------------------------------------------


def test_4synthlines_with_ngaussians():

  x = np.linspace(5800, 5803, num=1000)
  y = gaussian(x, -0.8, 5801.1, 0.2) + gaussian(x, -0.5, 5802.2, 0.2) + \
      gaussian(x, -0.3, 5801.7, 0.2) + gaussian(x, -0.2, 5802.8, 0.2) + \
      np.random.normal(0, 0.01, x.shape[0])

  params = [-0.9, 5801, 0.1, -0.4, 5802, 0.1,-0.4, 5801.5, 0.1, -0.1, 5803, 0.1]

  mod, out, init = lmfit_ngauss(x,y, params)

  
  plt.plot(x, y)
  plt.plot(x, init, 'k--')

  print(out.fit_report(min_correl=0.5))

  plt.plot(x, out.best_fit, 'r-')
  plt.show()



#-------------------------------------------------
#-------------------------------------------------

def test_4synthlines_with_ngaussians_atone():

  x = np.linspace(5800, 5803, num=1000)
  y = gaussian(x, -0.8, 5801.1, 0.2) + gaussian(x, -0.5, 5802.2, 0.2) + \
      gaussian(x, -0.3, 5801.7, 0.2) + gaussian(x, -0.2, 5802.8, 0.2) + \
      np.random.normal(0, 0.01, x.shape[0]) + 1.0

  params = [-0.9, 5801, 0.1, -0.4, 5802, 0.1,-0.4, 5801.5, 0.1, -0.1, 5803, 0.1]

  mod, out, init = lmfit_ngauss(x,y-1, params)

  
  plt.plot(x, y)
  plt.plot(x, init+1, 'k--')

  print(out.fit_report(min_correl=0.5))

  plt.plot(x, out.best_fit+1, 'r-')
  plt.show()


#-------------------------------------------------
#-------------------------------------------------


def test_4synthlines_with_ngaussians_constrains():

  x = np.linspace(5800, 5803, num=500)
  y = gaussian(x, -0.8, 5801.1, 0.2) + gaussian(x, -0.5, 5802.2, 0.2) + \
      gaussian(x, -0.3, 5801.7, 0.2) + gaussian(x, -0.2, 5802.7, 0.2) + \
      np.random.normal(0, 0.25, x.shape[0])

  params = [-0.9, 5801, 0.1, -0.4, 5802, 0.1,-0.4, 5801.5, 0.1, -0.1, 5802.8, 0.1]
  constrains = [(-1.,-0.1), (5800.8,5801.2), (0.05,0.3),(-1,-0.1), (5801.8,5802.2), (0.05,0.3),
                (-1.,-0.1), (5801.3,5801.8), (0.05,0.3),(-1,-0.1), (5802.6,5803), (0.05,0.3),]
  constrains = [(-2.,0.1), (5800.8,5801.2), (0.05,0.3),(-2,0.1), (5801.8,5802.2), (0.05,0.3),
                (-2.,0.1), (5801.3,5801.8), (0.05,0.3),(-2,0.1), (5802.6,5803), (0.05,0.3),]
  constrains = [(-2.,0.1), (5800,5802), (0.05,0.3),(-2,0.1), (5801,5802.5), (0.05,0.3),
                (-2.,0.1), (5801,5802), (0.05,0.3),(-2,0.1), (5802,5803.5), (0.05,0.3),]

  mod, out, init = lmfit_ngauss_constrains(x,y, params, constrains)

  
  plt.plot(x, y)
  plt.plot(x, init, 'k--')

  print(out.fit_report(min_correl=0.5))

  plt.plot(x, out.best_fit, 'r-')
  plt.show()






#-------------------------------------------------
#-------------------------------------------------

def test_2gaussians():

  x = np.linspace(0.0, 10.0, num=1000)
  y = gaussian(x, -1, 3, 0.75) + gaussian(x, -0.5, 5, 0.8) + np.random.normal(0, 0.01, x.shape[0])

  gauss1  = GaussianModel(prefix='g1_')
  
  pars = gauss1.guess(y, x=x)
  pars['g1_amplitude'].set(-0.9)
  pars['g1_center'].set(2.5)
  pars['g1_sigma'].set(0.5)

  gauss2  = GaussianModel(prefix='g2_')
  pars.update(gauss2.make_params())
  pars['g2_amplitude'].set(-0.4)
  pars['g2_center'].set(5)
  pars['g2_sigma'].set(0.5)

  mod = gauss1 + gauss2

  init = mod.eval(pars, x=x)

  plt.plot(x, y)
  plt.plot(x, init, 'k--')

  out = mod.fit(y, pars, x=x)

  print(out.fit_report(min_correl=0.5))

  plt.plot(x, out.best_fit, 'r-')
  plt.show()

#-------------------------------------------------
#-------------------------------------------------



def test_example_2_Gaussians_1_exp():
  dat = np.loadtxt('NIST_Gauss2.dat')
  x = dat[:, 1]
  y = dat[:, 0]

  exp_mod = ExponentialModel(prefix='exp_')
  pars = exp_mod.guess(y, x=x)

  gauss1  = GaussianModel(prefix='g1_')
  pars.update(gauss1.make_params())

  pars['g1_center'].set(105, min=75, max=125)
  pars['g1_sigma'].set(15, min=3)
  pars['g1_amplitude'].set(2000, min=10)

  gauss2  = GaussianModel(prefix='g2_')

  pars.update(gauss2.make_params())

  pars['g2_center'].set(155, min=125, max=175)
  pars['g2_sigma'].set(15, min=3)
  pars['g2_amplitude'].set(2000, min=10)

  mod = gauss1 + gauss2 + exp_mod


  init = mod.eval(pars, x=x)
  plt.plot(x, y)
  plt.plot(x, init, 'k--')

  out = mod.fit(y, pars, x=x)

  print(out.fit_report(min_correl=0.5))

  plt.plot(x, out.best_fit, 'r-')
  plt.show()
  #<end examples/doc_nistgauss.py>

#-------------------------------------------------
#-------------------------------------------------

## -> https://lmfit.github.io/lmfit-py/builtin_models.html#example-3-fitting-multiple-peaks-and-using-prefixes
#test_example_2_Gaussians_1_exp()


#test_2gaussians()


#test_2gaussians_with_ngaussians()
#test_2synthlines_with_ngaussians()

#some times work better than others due to the last line on the edge
#test_4synthlines_with_ngaussians()

#some times work better than others due to the last line on the edge
#test_4synthlines_with_ngaussians_atone()

#With constrains it works every time (never saw a bad fit)
test_4synthlines_with_ngaussians_constrains()



