from typing import Callable


class Model:
    def __init__(self, model_config, hyper_parameters) -> None:
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def evaluate(self, X, y):
        pass

    def save_weights(self, model_folder_path, **kwargs):
        pass

    def load_weights(cls, model_folder_path, **kwargs):
        pass

    def summary(self, fn: Callable[[str], None]) -> None:
        pass
