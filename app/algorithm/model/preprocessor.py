import os
from typing import List
import pandas as pd
import numpy as np


class Preprocessor:
    def __init__(self, preprocessor_config, data_schema, hyper_parameters):
        pass

    def fit(self, data: pd.DataFrame):
        pass

    def transform(self, data: pd.DataFrame, include_label: bool) -> dict:
        pass

    def label_to_class(self, labels: List[int]) -> List[str]:
        raise NotImplementedError(
            "label_to_class is not supported by {}".format(self.__class__.__name__))

    def num_classes(self):
        raise NotImplementedError(
            "num_classes is not supported by {}".format(self.__class__.__name__))

    def class_names(self) -> List[str]:
        raise NotImplementedError(
            "class_names is not supported by {}".format(self.__class__.__name__))

    def class_distribution(self, inverse_prob=True):
        raise NotImplementedError(
            "class_distribution is not supported by {}".format(self.__class__.__name__))

    def save_weights(self, model_folder_path, **kwargs):
        pass

    def load_weights(self, model_folder_path, **kwargs):
        pass

    def post_processing(self, output: np.ndarray) -> pd.DataFrame:
        pass


def load_or_fit(preprocessor: Preprocessor, model_path, data: pd.DataFrame):
    if os.path.exists(model_path):
        preprocessor.load_weights(model_path)
    else:
        preprocessor.fit(data)
        preprocessor.save_weights(model_path)
