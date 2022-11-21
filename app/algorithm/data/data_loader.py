import json
import os
import pandas as pd

from .data_schema import DataSchema


def get_json_file(file_path, file_type):
    try:
        json_data = json.load(open(file_path))
        return json_data
    except:
        raise Exception(f"Error reading {file_type} file at: {file_path}")


def get_hyperparameters(hyper_param_path):
    return get_json_file(hyper_param_path, "hyperparameters")


def get_data(data_path: str) -> pd.DataFrame:
    all_files = os.listdir(data_path)
    csv_files = list(filter(lambda f: f.endswith('.csv'), all_files))
    input_files = [os.path.join(data_path, file) for file in csv_files]
    if len(input_files) == 0:
        raise ValueError(f'There are no data files in {data_path}.')
    raw_data = [pd.read_csv(file) for file in input_files]
    data = pd.concat(raw_data)
    return data


def attach_test_labels(data: pd.DataFrame, test_labels_path: str, label_key: str) -> pd.DataFrame:
    test_labels = pd.read_csv(test_labels_path)
    data[label_key] = test_labels[label_key]
    return data


def get_data_schema(data_schema_path: str) -> DataSchema:
    try:
        json_files = list(filter(lambda f: f.endswith(
            '.json'), os.listdir(data_schema_path)))
        if len(json_files) > 1:
            raise Exception(
                f'Multiple json files found in {data_schema_path}. Expecting only one schema file.')
        full_fpath = os.path.join(data_schema_path, json_files[0])
        with open(full_fpath, 'r') as f:
            data_schema = json.load(f)
            return DataSchema(data_schema)
    except:
        raise Exception(
            f"Error reading data_schema file at: {data_schema_path}")


def save_predictions(predictions: pd.DataFrame, folder_path: str):
    predictions.to_csv(os.path.join(
        folder_path, 'test_predictions.csv'), index=False)
