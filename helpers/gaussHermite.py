'''
This file provides the computation of Gauss-Hermite quadrature points and weights. The functions rely on the numpy
implementation of the Gauss-Hermite quadrature points and weights. There is a univariate and multivariate Gauss-Hermite
quadrature function available. Still to be implemented: importance sampling version for bond-price function with more
weight on the default region.
'''
from numpy.polynomial.hermite import hermgauss
import numpy as np
import itertools

def gaussHermiteUnivariate(deg, stochProcess):
    """
    Univariate Gauss-Hermite quadrature

    Computes the corrected sample points and weights of the Gauss-Hermite quadrature given a stochastic process.

    Parameteres
    -----------
    deg : int
        Number of sample points and weights. Has to be >=1
    stochProcess : object
        Type of parent class stochProcess. See package processes

    Returns
    -------
    x : ndarray
        1-D array containing the corrected sample points
    w : ndarray
        1-D array containing the corrected weights (Correction term is division by sqrt(2))
    """
    # Get original points and weights
    pointsOrig, weightOrig = hermgauss(deg)
    volatility = stochProcess.getVolatility()

    # Correct points
    pointsCorrected = pointsOrig * np.sqrt(2) * volatility
    weightsCorrected = weightOrig / np.sqrt(np.pi)
    return pointsCorrected, weightsCorrected

def gaussHermiteMultivariate(deg, processes):
    nrOfDimensions = len(processes)
    const = np.pi ** (-0.5*nrOfDimensions)
    # Covariance matrix
    volVector = np.zeros(nrOfDimensions)
    for i in range(nrOfDimensions):
        volVector[i] = processes[i].getVolatility() ** 2
    Sigma = np.diag(volVector)
    # Gauss-Hermite Mdim
    x, w = hermgauss(deg)
    xn = np.array(list(itertools.product(*(x,)*nrOfDimensions)))
    wn = const * np.prod(np.array(list(itertools.product(*(w,) * nrOfDimensions))), 1)
    pointFinal = 2.0 ** 0.5 * np.dot(np.linalg.cholesky(Sigma), xn.T).T
    return pointFinal, wn