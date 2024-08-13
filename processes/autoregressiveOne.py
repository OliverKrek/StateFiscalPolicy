from processes.stochProcess import stochProcess
import numpy as np
# Implement the correlation via hascodes
class autoregressiveOne(stochProcess):
    def __init__(self, volatility, arPara, longRunMean, shockType='normal'):
        stochProcess.__init__(self)
        self.__shockType = shockType
        self.__arPara = arPara
        self.__volatility = volatility
        self.__longRunMean = longRunMean
        self.state = 0
        self.correlations ={}

    def simulate(self, N):
        processValues = np.zeros((N+1,1))
        processValues[0] = 0
        for i in range(1, N+1):
            processValues[i, :] = (self.__longRunMean + self.__arPara * processValues[i-1, :] +
                                   self.__volatility * np.random.randn(1))

        return processValues

    def draw(self, *args):
        if args:
            state = args[0]
        else:
            state = self.state
        newState = self.__longRunMean + self.__arPara * state + self.__volatility * np.random.randn(1)
        self.state = newState
        return newState

    def drawGH(self, state, shocks):
        newState = self.__longRunMean + self.__arPara * state + shocks
        self.state = newState
        return newState

    # Implement tensorflow versions of the function

    # Define getters and setters
    def setAR(self, arPara):
        self.__arPara = arPara

    def setVolatility(self, volatility):
        self.__volatility = volatility

    def setLongRunMean(self, longRunMean):
        self.__longRunMean = longRunMean

    def getAR(self):
        return self.__arPara

    def getVolatility(self):
        return self.__volatility

    def getLongRunMean(self):
        return self.__longRunMean
