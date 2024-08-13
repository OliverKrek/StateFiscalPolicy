'''
This file defines the class of a representative agent.
To Do's:
- add functionalities
- add saver function for fitted UC
'''
import numpy as np
import matplotlib.pyplot as plt
from helpers.gaussHermite import gaussHermiteUnivariate, gaussHermiteMultivariate


class RepAgent:
    def __init__(self, gamma=10, psi=1.5, betta=0.99):
        # public constructor. Takes parameters as arguments. Default values are set
        self.__gamma = gamma
        self.__psi = psi
        self.__betta = betta
        self.consProcess = None
        self.longRunRisk = None
        self.solverFlag = False
        self.ghFlag = False
        # Setup fields to be filled later

    ###### Solver for contraction mapping ######
    def initContractionSolver(self, nrOfPoints, initialValues, volScale=None, tolerance=None, metric=None):
        self.initGrid(nrOfPoints)
        self.initInterpolant(initialValues)
        self.initSolver()
        self.solverFlag = True

    def initGaussHermite(self, deg, *args):
        if self.solverFlag:
            # Univariate Gauss-Hermite
            ghPoints, ghWeights = gaussHermiteUnivariate(deg, self.longRunRisk)
            self.ghPoints = ghPoints
            self.ghWeights = ghWeights
            # Precomputation of x-Values for Gauss Hermite to vectorize contraction
            ghMV = np.zeros((self.xGrid.size, deg))
            for i in range(self.xGrid.size):
                ghMV[i, :] = self.longRunRisk.drawGH(self.xGrid[i], self.ghPoints)
            self.ghMV = ghMV.reshape(self.xGrid.size * deg)
            self.ghFlag = True

    # Set grid for linear interpolation
    def initGrid(self, nrOfPoints, volaScale = 3.5):
        lsdv = self.longRunRisk.getVolatility() / np.sqrt(1-self.longRunRisk.getAR() ** 2)
        self.xGrid = (
            np.linspace(- lsdv * volaScale, lsdv * volaScale, nrOfPoints))

    def initInterpolant(self, initialValues):
        # This function uses the numpy implementation of linear 1-d interpolation
        self.ucValues = initialValues

    def initSolver(self, tolerance=1e-6, metricRA=10):
        self.tolerance = tolerance
        self.metricRA = metricRA

    def contractionFunction(self):
        ucNext = np.interp(self.ghMV, self.xGrid, self.ucValues) ** (1-self.__gamma)
        ucNext = ucNext.reshape(self.xGrid.size, self.ghPoints.size)
        values = ( (1-self.__betta) + self.__betta * (np.exp(self.consProcess.getLongRunMean() + self.xGrid + 0.5 * (1-self.__gamma) * self.consProcess.getVolatility()**2)
                                                     * np.dot(ucNext, self.ghWeights) **(1/(1-self.__gamma))
                                                     ) ** (1-1/self.__psi) ) ** (1 / (1-1/self.__psi))
        return values

    def solveUC(self):
        # Add checkpoints that other function have to be invoked first
        if self.solverFlag and self.ghFlag:
            # Main loop
            i = 0
            while self.metricRA > self.tolerance:
                newUC = self.contractionFunction()
                self.metricRA = np.sqrt(np.sum(abs((newUC - self.ucValues)) ** 2))
                if i % 100 == 0:
                    print(f"Iteration: {i} ##### Error: {self.metricRA}")
                self.ucValues = newUC
                i += 1
            print(f"Success: Solver has Converged. ##### Error: {self.metricRA} ; Iteration: {i}")
        else:
            print("Need to setup the solver and Gauss-Hermite first")
    # Risk Free bond price
    def bondPriceRf(self, state, type = 'raw'):
        shocksBP, weightsBP = gaussHermiteMultivariate(10, [self.longRunRisk, self.consProcess])
        #xNextUC = self.longRunRisk.drawGH(state, self.ghPoints)
        xNextUC = self.longRunRisk.drawGH(state, shocksBP[:, 0])
        # Continuation Value UC
        contUC = np.dot(weightsBP, np.interp(xNextUC, self.xGrid, self.ucValues) ** (1-self.__gamma)) **(1/(1-self.__gamma))
        nextUC = np.interp(self.longRunRisk.drawGH(state, shocksBP[:, 0]), self.xGrid, self.ucValues)

        integrands = (self.__betta *
                      np.exp(-(1/self.__psi)*(self.consProcess.getLongRunMean()+state)
                             -self.__gamma*shocksBP[:,1]
                             -(1/self.__psi-self.__gamma) * 0.5 * (1-self.__gamma)*self.consProcess.getVolatility()**2) *
                      (nextUC/contUC) ** (1/self.__psi-self.__gamma)
                      )
        if type.lower() == 'yield':
            bondPrice = np.dot(integrands, weightsBP)
            return np.exp(-np.log(bondPrice))-1
        else:
            return np.dot(integrands, weightsBP)




    # Getters and Setters
    def setGamma(self, gamma):
        self.__gamma = gamma

    def setPsi(self, psi):
        self.__psi = psi

    def setBetta(self, betta):
        self.__betta = betta

    def getGamma(self):
        return self.__gamma

    def getPsi(self):
        return self.__psi

    def getBetta(self):
        return self.__betta

    # Set stochastic processes
    def setConsProcess(self, consProcess):
        self.consProcess = consProcess

    def setLongRunRisk(self, longRunRisk):
        self.longRunRisk = longRunRisk