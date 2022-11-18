def get_or_def(obj, key, default):
    if key in obj:
        return obj[key]
    else:
        return default
