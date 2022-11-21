import os

from transformers import GPT2ForSequenceClassification
from algorithm.misc.utils import verify_folder


def init_model(folder_path, **kwargs):
    verify_folder(folder_path)

    model_path = os.path.join(folder_path, kwargs['model_file_name'])
    if not os.path.exists(model_path):
        model = GPT2ForSequenceClassification.from_pretrained('gpt2')
        model.save_pretrained(model_path)
        print('Downloaded model for GPT2 at {}'.format(model_path))
    else:
        print('Using pre-trained model for GPT2 at {}'.format(model_path))
