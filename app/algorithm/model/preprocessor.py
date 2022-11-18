import os
import pandas as pd


class Preprocessor:
    def __init__(self, data_schema, hyper_parameters):
        pass

    def fit(self, data: pd.DataFrame):
        pass

    def transform(self, data: pd.DataFrame) -> dict:
        pass

    def save_weights(self, model_path):
        pass

    def load_weights(self, model_path):
        pass


def load_or_fit(preprocessor: Preprocessor, model_path, data: pd.DataFrame):
    if os.path.exists(model_path):
        preprocessor.load_weights(model_path)
    else:
        preprocessor.fit(data)
        preprocessor.save_weights(model_path)
