import numpy as np

from typing import Callable


class Model:
    def __init__(self, model_config, hyper_parameters) -> None:
        """
        abstract class for ML model

        :param model_config: model config
        :param hyper_parameters: hyper parameters
        """
        pass

    def model_name(self) -> str:
        """
        get the name of this model
        """
        pass

    def fit(self, X, y):
        """
        fit the model to the data.

        :param X: input data. The type or shape of X depends on the model.
        :param y: target data. The type or shape of y depends on the model.
        """
        pass

    def predict(self, X) -> np.ndarray:
        """
        predict label for input data X

        :param X: input data. The type or shape of X depends on the model.
        :return: predicted label
        """
        pass

    def evaluate(self, X, y) -> dict:
        """
        evaluate the model performance on the data.

        :param X: the 'test' input data.
        :param y: the 'test' target data.
        :return: the evaluation result. 
        """
        pass

    def save_weights(self, model_folder_path, **kwargs):
        """
        save the model weights and any relevant state to the model_folder_path

        :param model_folder_path: the folder path to save the model weights
        :param kwargs: any additional arguments
        """
        pass

    def load_weights(cls, model_folder_path, **kwargs):
        """
        load the saved stated (done by save_weights) from the model_folder_path

        :param model_folder_path: the folder path to load the model weights
        :param kwargs: any additional arguments
        """

        pass

    def summary(self, fn: Callable[[str], None]) -> None:
        """
        print the summary of the model. Includes information such as the model architecture.

        :param fn: a function that takes a string as input and print the string. The string
        passed to fn should be the summary of the model.
        """
        pass
