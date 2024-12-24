from abc import ABC, abstractmethod


class ExtractHandler(ABC):

    @abstractmethod
    def extract(self):
        pass
