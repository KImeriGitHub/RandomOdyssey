from abc import ABC, abstractmethod

class ISimulation(ABC):
    @abstractmethod
    def run(self):
        pass