'''
This file contains the parameters for the underlying economic model. It is the single source of the deep structural
values used in all calculations. Please modify parameter values only in the file.
'''
import numpy as np
# Economic parameters
# Relative risk aversion
gamma = 11
gammaG = 5.
# IES
psi = 1.5
psiG = 2
# Discount factors
betta = 0.997
bettaG = 0.9
# average growth
xBar = 0.006
# Volatility consumption
sigC = 0.008
# Volatility scale long run component
phiE = 0.13
# Persistence of long run component
rhoX = 0.982
corrXZ = 0.9
# Volatility long run component
sigX = phiE*sigC
stdX = sigX/np.sqrt(1-rhoX**2)
# Standard deviation of transitory income shocks
sigZ = 0.012
# Persistence of transitory shocks
rhoZ = 0.983
stdZ = sigZ / np.sqrt(1-rhoZ**2)
# Leverage long-run shocks
phi = 3
# Probability of getting good credit rating
theta = 0.1
# Consumption cost in case of default
delta = 0.3
# Recovery rate in case of default
rRate = 0.5
# Tax rule
rhoT = 0.2
kappa = 0.03
varPhi = 0.7
tauBar = 0.2
debtBar = 0.15
sStar = 0.005
outB = 1. * np.exp(xBar)
# Adj. Costs
largeTheta = 5.5
largePhi = 42.5
# Agency cost of holding cash
deltaI = 0.9
# Debt maturity
lambdaD = 0.2