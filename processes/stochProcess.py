"""
This is an abstract base class file for all types of stochastic processes.
"""
from abc import ABC, abstractmethod

class stochProcess(ABC):
    @abstractmethod
    def __init__(self, shockType='normal'):
        super().__init__()
        self.shockType = shockType

    @abstractmethod
    def simulate(self, N):
        pass

    @abstractmethod
    def draw(self):
        pass

