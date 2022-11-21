import os

from transformers import GPT2Tokenizer
from algorithm.misc.utils import verify_folder


def init_preprocessor(folder_path, **kwargs):
    verify_folder(folder_path)

    tokenizer_path = os.path.join(folder_path, kwargs['tokenizer_file_name'])
    if not os.path.exists(tokenizer_path):
        preprocessor = GPT2Tokenizer.from_pretrained('gpt2')
        preprocessor.save_pretrained(tokenizer_path)
        print('Downloaded tokenizer for GPT2 at {}'.format(tokenizer_path))
    else:
        print('Using pre-trained tokenizer for GPT2 at {}'.format(tokenizer_path))
