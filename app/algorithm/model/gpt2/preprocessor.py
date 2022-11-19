import numpy
import pandas as pd
import torch

from transformers import GPT2Tokenizer
from sklearn.preprocessing import LabelEncoder

from ..preprocessor import Preprocessor
from ...config import *
from ...utils import get_or_def, verify_folder


TEXT_COL_KEY = 'documentField'
Y_COL_KEY = 'targetField'


def init_preprocessor(folder_path, **kwargs):
    verify_folder(folder_path)

    tokenizer_path = os.path.join(folder_path, kwargs['tokenizer_file_name'])
    if not os.path.exists(tokenizer_path):
        preprocessor = GPT2Tokenizer.from_pretrained('gpt2')
        preprocessor.save_pretrained(tokenizer_path)
        print('Downloaded tokenizer for GPT2 at {}'.format(tokenizer_path))
    else:
        print('Using pre-trained tokenizer for GPT2 at {}'.format(tokenizer_path))


class GPT2Preprocessor(Preprocessor):
    def __init__(self, preprocessor_config, data_schema, hyper_parameters):
        self.data_spec = data_schema['inputDatasets'][SCHEMA_TYPE]
        self.batch_size = get_or_def(hyper_parameters, 'batch_size', 4)
        self.max_length = get_or_def(hyper_parameters, 'max_length', 64)
        self.sample_size = get_or_def(hyper_parameters, 'sample_size', 0)
        self.padding_side = get_or_def(
            hyper_parameters, 'padding_side', 'right')

        self.tokenizer = GPT2Tokenizer.from_pretrained(
            preprocessor_config['tokenizer_path'], local_files_only=True)
        self.label_encoder = LabelEncoder()

        self.tokenizer.pad_token = '[PAD]'

    def fit(self, data: pd.DataFrame):
        self.label_encoder.fit(data[self.data_spec[Y_COL_KEY]])
        self.label_distribution = data[self.data_spec[Y_COL_KEY]].value_counts(
            normalize=True).to_dict()

    def _sample(self, data: pd.Series):
        if self.sample_size < len(data) and self.sample_size > 0:
            return data.sample(self.sample_size, replace=False)
        else:
            return data

    def transform(self, data) -> dict:
        X = data[self.data_spec[TEXT_COL_KEY]]
        X = self._sample(X)
        y = data[self.data_spec[Y_COL_KEY]]
        y = self._sample(y)

        # {input_ids: N X MAX_LENGTH, attention_mask: N X 1}
        X_dict = self.tokenizer(X.tolist(),
                                padding='max_length',
                                truncation=True,
                                max_length=self.max_length,
                                return_tensors='pt')

        y = self.label_encoder.transform(y)
        y = torch.tensor(y)
        return {'X': X_dict, 'y': y}

    def label_to_class(self, labels):
        return self.label_encoder.inverse_transform(labels)

    def num_classes(self):
        return len(self.label_encoder.classes_)

    def class_distribution(self, inverse_prob=True):
        probs = list(map(self.label_distribution.get,
                     self.label_encoder.classes_))
        if inverse_prob:
            probs = [1 - p for p in probs]
        return probs

    def save_weights(self, model_folder_path, **kwargs):
        verify_folder(model_folder_path)

        # tokenizer_path = os.path.join(
        #     model_folder_path, kwargs['tokenizer_file_name'])
        # self.tokenizer.save_pretrained(tokenizer_path)
        # print("Tokenizer saved to {}".format(tokenizer_path))

        label_encoder_path = os.path.join(
            model_folder_path, kwargs['label_encoder_file_name'])
        numpy.save(label_encoder_path, self.label_encoder.classes_)
        print("Label encoder saved to {}".format(label_encoder_path))

    def load_weights(self, model_folder_path, **kwargs):
        verify_folder(model_folder_path)

        # tokenizer_path = os.path.join(
        #     model_folder_path, kwargs['tokenizer_file_name'])
        # if os.path.exists(tokenizer_path):
        #     self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        #     print("Tokenizer loaded from {}".format(tokenizer_path))

        label_encoder_path = os.path.join(
            model_folder_path, kwargs['label_encoder_file_name'])
        if os.path.exists(label_encoder_path):
            self.label_encoder.classes_ = numpy.load(
                label_encoder_path, allow_pickle=True)
            print("Label encoder loaded from {}".format(label_encoder_path))
