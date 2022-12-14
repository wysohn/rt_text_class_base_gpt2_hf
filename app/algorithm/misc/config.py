import os

VERSION = '0.0.1'

BASE_FOLDER_PATH = os.getenv('BASE_PATH', '/opt/ml_vol')
CHANNEL = os.getenv('CHANNEL', 'training')
SCHEMA_TYPE = os.getenv('SCHEMA_TYPE', 'textClassificationBaseMainInput')
EVALUATE_FILE = os.getenv('EVALUATE_FILE', None)

print('Settings: {}'.format({
    'VERSION': VERSION,
    'BASE_FOLDER_PATH': BASE_FOLDER_PATH,
    'CHANNEL': CHANNEL,
    'SCHEMA_TYPE': SCHEMA_TYPE
}))

INPUT_FOLDER_PATH = os.path.join(BASE_FOLDER_PATH, 'inputs')
INPUT_DATA_FOLDER_PATH = os.path.join(
    INPUT_FOLDER_PATH, 'data', CHANNEL, SCHEMA_TYPE)
INPUT_DATA_CONFIG_FOLDER_PATH = os.path.join(INPUT_FOLDER_PATH, 'data_config')
INPUT_DATA_TEST_KEY_FOLDER_PATH = os.path.join(INPUT_DATA_FOLDER_PATH, 'keys')

OUTPUT_FOLDER_PATH = os.path.join(BASE_FOLDER_PATH, 'outputs')
OUTPUT_TESTING_FOLDER_PATH = os.path.join(
    OUTPUT_FOLDER_PATH, 'testing_outputs')
OUTPUT_ERROR_FOLDER_PATH = os.path.join(OUTPUT_FOLDER_PATH, 'errors')
OUTPUT_ERROR_TRAIN_FAILURE_FILE_PATH = os.path.join(
    OUTPUT_ERROR_FOLDER_PATH, 'train_failure')
OUTPUT_ERROR_TEST_FAILURE_FILE_PATH = os.path.join(
    OUTPUT_ERROR_FOLDER_PATH, 'test_failure')
OUTPUT_ERROR_SERVING_FAILURE_FILE_PATH = os.path.join(
    OUTPUT_ERROR_FOLDER_PATH, 'serving_failure')

MODEL_FOLDER_PATH = os.path.join(BASE_FOLDER_PATH, 'model')
MODEL_CONFIG_FOLDER_PATH = os.path.join(MODEL_FOLDER_PATH, 'model_config')
MODEL_HYPERPARAMETERS_FILE_PATH = os.path.join(
    MODEL_CONFIG_FOLDER_PATH, 'hyperparameters.json')
MODEL_ARTIFACT_FOLDER_PATH = os.path.join(MODEL_FOLDER_PATH, 'artifacts')
MODEL_HISTORY_FILE_PATH = os.path.join(
    MODEL_ARTIFACT_FOLDER_PATH, 'history.json')

PRETRAINED_MODEL_FOLDER_PATH = '/models'

# Internal configs

MODEL_SAVE_CONFIG = {
    'model_file_name': 'gpt2_model'
}
PREPROCESSOR_SAVE_CONFIG = {
    'tokenizer_file_name': 'gpt2_tokenizer',
    'label_encoder_file_name': 'label_encoder.npy',
    'label_distribution_file_name': 'label_distribution.npy'
}
