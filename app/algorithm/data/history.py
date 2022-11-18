import os
import pandas as pd

from ..config import *


class History:
    """
    Simple copy of the keras history object.
    """

    def __init__(self):
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


def save_training_history(history: History):
    hist_df = pd.DataFrame(history.history)
    with open(MODEL_HISTORY_FILE_PATH, mode='w') as f:
        hist_df.to_json(f)
