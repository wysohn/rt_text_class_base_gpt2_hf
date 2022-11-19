from algorithm.model import Model, Preprocessor, init_model, init_preprocessor
import pandas as pd
import torch

from algorithm.data import data_loader

MODEL_ARTIFACT_FOLDER_PATH = 'local_test/ml_vol/model/artifacts'
INPUT_DATA_FOLDER_PATH = 'local_test/ml_vol/inputs/data/testing/textClassificationBaseMainInput'

if __name__ == '__main__':
    hyper_parameters = {
        'batch_size': 2,
        'max_length': 64,
        'epoch': 32,
        'sample_size': 1000,
    }

    init_model('local_test/model', model_file_name='gpt2_pretrained')
    init_preprocessor('local_test/model', **{
        'tokenizer_file_name': 'gpt2_tokenizer',
        'label_encoder_file_name': 'gpt2_label_encoder.npy',
    })

    data = data_loader.get_data(INPUT_DATA_FOLDER_PATH)

    data_schema = {
        "problemCategory": "text_classification_base",
        "version": "1.0",
        "language": "en-us",
        "encoding": "utf-8",
        "inputDatasets": {
            "textClassificationBaseMainInput": {
                "idField": "Id",
                "targetField": "Category",
                "documentField": "Message"
            }
        }}

    preprocessor = Preprocessor({
        'tokenizer_path': 'local_test/model/gpt2_tokenizer',
    }, data_schema, hyper_parameters)

    model_save_config = {
        'model_file_name': 'gpt2_model'
    }
    preprocessor_save_config = {
        'tokenizer_file_name': 'gpt2_tokenizer',
        'label_encoder_file_name': 'label_encoder.npy'
    }

    preprocessor = Preprocessor({
        'tokenizer_path': 'local_test/model/gpt2_tokenizer'
    }, data_schema, hyper_parameters)

    preprocessor.load_weights(
        MODEL_ARTIFACT_FOLDER_PATH, **preprocessor_save_config)

    # Prepare model
    model = Model({
        'model_path': 'local_test/model/gpt2_pretrained',
        'num_labels': preprocessor.num_classes(),
        'class_weights': preprocessor.class_distribution(),
    }, hyper_parameters)

    model.load_weights(MODEL_ARTIFACT_FOLDER_PATH, **model_save_config)

    # Preprocess raw test data
    processed_data_dict = preprocessor.transform(data)
    # evaluate model
    print("loss: {}".format(model.evaluate(**processed_data_dict)))
