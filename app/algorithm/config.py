import os

BASE_FOLDER_PATH = os.getenv('BASE_PATH', '/opt/ml_vol')
CHANNEL = os.getenv('CHANNEL', 'training')
SCHEMA_TYPE = os.getenv('SCHEMA_TYPE', 'textClassificationBaseMainInput')

print('Settings: {}'.format({
    'BASE_FOLDER_PATH': BASE_FOLDER_PATH,
    'CHANNEL': CHANNEL,
    'SCHEMA_TYPE': SCHEMA_TYPE
}))

INPUT_FOLDER_PATH = os.path.join(BASE_FOLDER_PATH, 'inputs')
INPUT_DATA_FOLDER_PATH = os.path.join(
    INPUT_FOLDER_PATH, 'data', CHANNEL, SCHEMA_TYPE)
INPUT_DATA_CONFIG_FOLDER_PATH = os.path.join(INPUT_FOLDER_PATH, 'data_config')

OUTPUT_FOLDER_PATH = os.path.join(BASE_FOLDER_PATH, 'outputs')
OUTPUT_ERROR_FOLDER_PATH = os.path.join(OUTPUT_FOLDER_PATH, 'errors')
OUTPUT_ERROR_TRAIN_FAILURE_FILE_PATH = os.path.join(
    OUTPUT_ERROR_FOLDER_PATH, 'train_failure')
OUTPUT_ERROR_TEST_FAILURE_FILE_PATH = os.path.join(
    OUTPUT_ERROR_FOLDER_PATH, 'test_failure')

MODEL_FOLDER_PATH = os.path.join(BASE_FOLDER_PATH, 'model')
MODEL_CONFIG_FOLDER_PATH = os.path.join(MODEL_FOLDER_PATH, 'model_config')
MODEL_HYPERPARAMETERS_FILE_PATH = os.path.join(
    MODEL_CONFIG_FOLDER_PATH, 'hyperparameters.json')
MODEL_ARTIFACT_FOLDER_PATH = os.path.join(MODEL_FOLDER_PATH, 'artifacts')
MODEL_HISTORY_FILE_PATH = os.path.join(
    MODEL_ARTIFACT_FOLDER_PATH, 'history.json')

PRETRAINED_MODEL_FOLDER_PATH = '/models'
