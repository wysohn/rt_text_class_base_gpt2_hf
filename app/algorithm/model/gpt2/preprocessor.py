import pandas as pd
import torch

from transformers import GPT2Tokenizer
from sklearn.preprocessing import LabelEncoder

from ..preprocessor import Preprocessor
from ...config import *
from ...utils import get_or_def


TEXT_COL_KEY = 'documentField'
Y_COL_KEY = 'targetField'


class GPT2Preprocessor(Preprocessor):
    def __init__(self, data_schema, hyper_parameters):
        super().__init__(data_schema, hyper_parameters)
        self.data_spec = data_schema['inputDatasets'][SCHEMA_TYPE]
        self.batch_size = get_or_def(hyper_parameters, 'batch_size', 4)
        self.max_length = get_or_def(hyper_parameters, 'max_length', 64)
        self.padding_side = get_or_def(
            hyper_parameters, 'padding_side', 'right')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.label_encoder = LabelEncoder()

        self.tokenizer.pad_token = '[PAD]'

    def fit(self, data: pd.DataFrame):
        print("Using pre-trained tokenizer for GPT2")
        self.label_encoder.fit(data[self.data_spec[Y_COL_KEY]])

    def transform(self, data) -> dict:
        X = data[self.data_spec[TEXT_COL_KEY]].tolist()
        y = data[self.data_spec[Y_COL_KEY]]

        # {input_ids: N X MAX_LENGTH, attention_mask: N X 1}
        X_dict = self.tokenizer(X,
                                padding='max_length',
                                truncation=True,
                                max_length=self.max_length,
                                return_tensors='pt')

        y = self.label_encoder.transform(y)
        y = torch.tensor(y)
        return {'X': X_dict, 'y': y}

    def save_weights(self, model_path):
        self.tokenizer.save_pretrained(model_path)

    def load_weights(self, model_path):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
