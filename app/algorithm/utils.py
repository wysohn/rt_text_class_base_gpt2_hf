import os


def get_or_def(obj, key, default):
    if key in obj:
        return obj[key]
    else:
        return default


def verify_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    assert os.path.isdir(folder_path), f'{folder_path} is not a directory'
