from typing import Callable


class Model:
    def __init__(self, hyper_parameters) -> None:
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def evaluate(self, x_test, y_test):
        pass

    def save_weights(self, model_path):
        pass

    def load_weights(cls, model_path):
        pass

    def summary(self, fn: Callable[[str], None]) -> None:
        pass
