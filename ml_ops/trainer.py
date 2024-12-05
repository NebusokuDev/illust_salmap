from abc import ABC, abstractmethod


class TrainerBase(ABC):
    @abstractmethod
    def _training_step(self):
        pass

    @abstractmethod
    def _testing(self):
        pass

    @abstractmethod
    def _training(self):
        for epoch in range(100):
            pass

    def save_model(self):
        pass

    def fit(self):
        self._training()
        self._testing()
        self.save_model()
