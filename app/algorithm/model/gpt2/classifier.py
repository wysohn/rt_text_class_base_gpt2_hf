import os
from typing import Callable
import torch
from transformers import GPT2ForSequenceClassification
from torch import stack
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.metrics import confusion_matrix

from ...data.history import History
from ...utils import get_or_def, verify_folder
from ..model import Model


def init_model(folder_path, **kwargs):
    verify_folder(folder_path)

    model_path = os.path.join(folder_path, kwargs['model_file_name'])
    if not os.path.exists(model_path):
        model = GPT2ForSequenceClassification.from_pretrained('gpt2')
        model.save_pretrained(model_path)
        print('Downloaded model for GPT2 at {}'.format(model_path))
    else:
        print('Using pre-trained model for GPT2 at {}'.format(model_path))


class GPT2ModelWrapper(Model):
    def __init__(self, model_config, hyper_parameters):
        self.batch_size = get_or_def(hyper_parameters, 'batch_size', 4)
        self.epoch = get_or_def(hyper_parameters, 'epoch', 32)
        self.class_weights = get_or_def(model_config, 'class_weights', None)
        self.num_labels = get_or_def(model_config, 'num_labels', 2)
        assert self.class_weights is None or len(
            self.class_weights) == self.num_labels

        self.model = GPT2ForSequenceClassification.from_pretrained(
            model_config['model_path'],
            local_files_only=True,
            num_labels=model_config['num_labels'])
        self.model.config.pad_token_id = self.model.config.eos_token_id

    def _batch_X(self, X):
        return [{'input_ids': input_ids_batch, 'attention_mask': attention_mask_batch}
                for input_ids_batch, attention_mask_batch
                in zip(self._batch(X['input_ids']), self._batch(X['attention_mask']))]

    def _batch(self, data):
        return [data[i:i + self.batch_size] for i in range(0, len(data), self.batch_size)]

    def fit(self, X, y):
        X = self._batch_X(X)
        y = self._batch(y)
        assert len(X) == len(y), '{}, {}'.format(X, y)

        print("class weights: {}".format(self.class_weights))

        optimizer = Adam(self.model.parameters(), lr=1e-5)
        loss_fct = CrossEntropyLoss(weight=torch.tensor(
            self.class_weights) if self.class_weights else None)
        history = History()
        for epoch in range(self.epoch):
            loss_sum = 0
            iterations = 0
            for batch_x, batch_y in zip(X, y):
                pred = self.model(**batch_x)
                loss = loss_fct(pred.logits, batch_y)

                loss.backward()
                optimizer.step()

                loss_sum += loss.item()
                iterations += 1

            epoch_loss = loss_sum / iterations
            history.on_epoch_end(epoch, {'loss': epoch_loss})
            print(f'Epoch {epoch} loss: {epoch_loss}')

        return history

    def predict(self, input_ids, attention_mask):
        classifier_output = self.model(input_ids=input_ids,
                                       attention_mask=attention_mask)
        return classifier_output.logits.detach().numpy()

    def evaluate(self, X, y):
        y_true = y
        y_pred = []

        X = self._batch_X(X)
        y = self._batch(y)
        assert len(X) == len(y), '{}, {}'.format(X, y)

        loss_sum = 0
        iterations = 0

        for batch_x, batch_y in zip(X, y):
            with torch.no_grad():
                pred = self.model(**batch_x, labels=batch_y)
            loss = pred.loss
            y_pred.extend(pred.logits.argmax(dim=-1).detach().numpy())

            loss_sum += loss.item()
            iterations += 1

        return loss_sum / iterations, confusion_matrix(y_true, y_pred)

    def save_weights(self, model_folder_path, **kwargs):
        verify_folder(model_folder_path)

        model_path = os.path.join(model_folder_path, kwargs['model_file_name'])
        self.model.save_pretrained(model_path)
        print('Saved model to {}'.format(model_path))

    def load_weights(self, model_folder_path, **kwargs):
        verify_folder(model_folder_path)

        model_path = os.path.join(model_folder_path, kwargs['model_file_name'])
        if os.path.exists(model_path):
            self.model = GPT2ForSequenceClassification.from_pretrained(
                model_path)
            print('Loaded model from {}'.format(model_path))

    def summary(self, fn=print):
        return fn(self.model)
