import numpy
import pandas as pd
import torch
import numpy as np

from transformers import GPT2Tokenizer
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax

from ..preprocessor import Preprocessor
from ...config import *
from ...utils import get_or_def, verify_folder
from ...data.data_schema import DataSchema


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
    def __init__(self, preprocessor_config, data_schema: DataSchema, hyper_parameters):
        self.data_schema = data_schema
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
        self.label_encoder.fit(data[self.data_schema.col_label_key()])
        self.label_distribution = data[self.data_schema.col_label_key()].value_counts(
            normalize=True).to_dict()

    def _sample(self, data: pd.Series):
        if self.sample_size < len(data) and self.sample_size > 0:
            return data.sample(self.sample_size, replace=False)
        else:
            return data

    def transform(self, data, include_label=True) -> dict:
        X = data[self.data_schema.col_text_key()]
        X = self._sample(X)
        if include_label:
            y = data[self.data_schema.col_label_key()]
            y = self._sample(y)

            y = self.label_encoder.transform(y)
            y = torch.tensor(y)

        # {input_ids: N X MAX_LENGTH, attention_mask: N X 1}
        X_dict = self.tokenizer(X.tolist(),
                                padding='max_length',
                                truncation=True,
                                max_length=self.max_length,
                                return_tensors='pt')

        out = {'X': X_dict}
        if include_label:
            out['y'] = y
        return out

    def label_to_class(self, labels):
        return self.label_encoder.inverse_transform(labels)

    def class_names(self):
        return self.label_encoder.classes_.tolist()

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

        label_distribution_path = os.path.join(
            model_folder_path, kwargs['label_distribution_file_name'])
        numpy.save(label_distribution_path, self.label_distribution)
        print("Label distribution saved to {}".format(label_distribution_path))

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

        label_distribution_path = os.path.join(
            model_folder_path, kwargs['label_distribution_file_name'])
        if os.path.exists(label_distribution_path):
            self.label_distribution = numpy.load(
                label_distribution_path, allow_pickle=True).item()
            print("Label distribution loaded from {}".format(
                label_distribution_path))

    def post_processing(self, output: np.ndarray) -> pd.DataFrame:
        classes = self.class_names()
        output = softmax(output, axis=-1)
        return pd.DataFrame(output, columns=classes)
