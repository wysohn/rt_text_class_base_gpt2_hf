import os
from typing import List
import pandas as pd
import numpy as np


class Preprocessor:
    def __init__(self, preprocessor_config, data_schema, hyper_parameters):
        """
        abstract class for preprocessor

        :param preprocessor_config: preprocessor config
        :param data_schema: data schema
        :param hyper_parameters: hyper parameters
        """
        pass

    def fit(self, data: pd.DataFrame):
        """
        fit the preprocessor to the data. May not have any effect for some preprocessors.

        :param data: input data. The type or shape of data depends on the preprocessor.
        """
        pass

    def transform(self, data: pd.DataFrame, include_label: bool) -> dict:
        """
        transform the data to the format that the model can accept.

        :param data: input data. The type or shape of data depends on the preprocessor.
        :param include_label: whether to include the label in the output. Useful if
            the preprocessor is used for inference.
        """
        pass

    def label_to_class(self, labels: List[int]) -> List[str]:
        """
        convert integer labels to class names

        :param labels: integer labels
        :return: class names
        """
        raise NotImplementedError(
            "label_to_class is not supported by {}".format(self.__class__.__name__))

    def num_classes(self):
        """
        get the number of classes

        :return: the number of classes
        """
        raise NotImplementedError(
            "num_classes is not supported by {}".format(self.__class__.__name__))

    def class_names(self) -> List[str]:
        """
        get the class names

        :return: the class names
        """
        raise NotImplementedError(
            "class_names is not supported by {}".format(self.__class__.__name__))

    def class_distribution(self, inverse_prob=True):
        """
        get the relative frequency of each class in the training data.
        The result is in exact same order as the class names.

        :param inverse_prob: whether to return the inverse probability (1 - p)
        """
        raise NotImplementedError(
            "class_distribution is not supported by {}".format(self.__class__.__name__))

    def save_weights(self, model_folder_path, **kwargs):
        """
        save the preprocessor weights and any relevant state to the model_folder_path

        :param model_folder_path: the folder path to save the preprocessor weights
        """
        pass

    def load_weights(self, model_folder_path, **kwargs):
        """
        load the preprocessor weights and any relevant state from the model_folder_path

        :param model_folder_path: the folder path to load the preprocessor weights
        """
        pass

    def post_processing(self, output: np.ndarray) -> pd.DataFrame:
        """
        post processing the output of the model. This may be removed in the future and 
        replaced by a more general post processing mechanism.

        :param output: the output of the model
        """
        pass


def load_or_fit(preprocessor: Preprocessor, model_path, data: pd.DataFrame):
    if os.path.exists(model_path):
        preprocessor.load_weights(model_path)
    else:
        preprocessor.fit(data)
        preprocessor.save_weights(model_path)
