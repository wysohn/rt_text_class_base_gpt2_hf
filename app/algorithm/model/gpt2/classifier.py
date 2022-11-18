from typing import Callable
from transformers import GPT2ForSequenceClassification
from torch import stack
from torch.optim import Adam

from ...data.history import History
from ...utils import get_or_def
from ..model import Model


class GPT2ModelWrapper(Model):
    def __init__(self, hyper_parameters):
        self.batch_size = get_or_def(hyper_parameters, 'batch_size', 4)
        self.epoch = get_or_def(hyper_parameters, 'epoch', 32)

        self.model = GPT2ForSequenceClassification.from_pretrained('gpt2')
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

        optimizer = Adam(self.model.parameters(), lr=1e-5)
        history = History()
        for epoch in range(self.epoch):
            loss_sum = 0
            iterations = 0
            for batch_x, batch_y in zip(X, y):
                pred = self.model(**batch_x, labels=batch_y)
                loss = pred.loss

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

    def evaluate(self, x_test, y_test):
        pass

    def save_weights(self, model_path):
        self.model.save_pretrained(model_path)

    def load_weights(self, model_path):
        self.model = GPT2ForSequenceClassification.from_pretrained(model_path)

    def summary(self, fn=print):
        return fn(self.model)
