from algorithm.model import Model, Preprocessor
import pandas as pd
import torch

if __name__ == '__main__':
    hyper_parameters = {
        'batch_size': 2,
        'max_length': 64,
        'epoch': 32,
        'sample_size': 4,
    }

    model = Model(hyper_parameters)

    preprocessor = Preprocessor({
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
        }}, {})

    df = pd.DataFrame({
        'Id': [1, 2, 3, 4],
        'Category': ['a', 'b', 'a', 'b'],
        'Message': ["Replace me by any text you'd like.", "Other sample",
                    "Replace me by any text you'd like.", "Other sample"]
    })

    preprocessor.fit(df)
    out = preprocessor.transform(df)

    model.fit(**out)
