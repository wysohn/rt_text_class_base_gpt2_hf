# major part of code sourced from aws sagemaker example:
# https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/scikit_bring_your_own/container/decision_trees/predictor.py

from algorithm.config import *
from algorithm import Model, Preprocessor, data_loader

import algorithm.utils as utils
import io
import pandas as pd
import json
import numpy as np
import flask
import traceback
import sys
import os
import warnings
from scipy.special import softmax
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore')

# Load configuarations and data
hyper_parameters = data_loader.get_hyperparameters(
    MODEL_HYPERPARAMETERS_FILE_PATH)
data_schema = data_loader.get_data_schema(
    INPUT_DATA_CONFIG_FOLDER_PATH)

# Prepare preprocessor
preprocessor = Preprocessor({
    'tokenizer_path': os.path.join(PRETRAINED_MODEL_FOLDER_PATH, 'gpt2_tokenizer')
}, data_schema, {**hyper_parameters, **{'sample_size': 0}})  # for serving, must use all samples

# load preprocessor saved data
preprocessor.load_weights(
    MODEL_ARTIFACT_FOLDER_PATH, **PREPROCESSOR_SAVE_CONFIG)

# Prepare model
model = Model({
    'model_path': os.path.join(PRETRAINED_MODEL_FOLDER_PATH, 'gpt2_pretrained'),
    'num_labels': preprocessor.num_classes(),
}, hyper_parameters)

model.load_weights(MODEL_ARTIFACT_FOLDER_PATH, **MODEL_SAVE_CONFIG)

# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. """
    status = 200
    response = f"Hello - I am {model.get_name()} model and I am at your service!"
    print(response)
    return flask.Response(response=response, status=status, mimetype="application/json")


@app.route("/infer", methods=["POST"])
def infer():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == "text/csv":
        data = flask.request.data.decode("utf-8")
        s = io.StringIO(data)
        data = pd.read_csv(s)
    else:
        return flask.Response(
            response="This predictor only supports CSV data",
            status=415, mimetype="text/plain"
        )

    print(f"Invoked with {data.shape[0]} records")

    # Do the prediction
    try:
        # Preprocess raw test data
        processed_data_dict = preprocessor.transform(data, False)

        # Make predictions
        predictions = model.predict(processed_data_dict['X'])
        predictions = preprocessor.post_processing(predictions)

        # Append ids to predictions
        predictions.insert(0, data_schema.col_id_key(),
                           processed_data_dict['ids'])

        # Convert from dataframe to CSV
        out = io.StringIO()
        predictions.to_csv(out, index=False)
        result = out.getvalue()

        return flask.Response(response=result, status=200, mimetype="text/csv")

    except Exception as err:
        # Write out an error file. This will be returned as the failureReason to the client.
        trc = traceback.format_exc()
        with open(OUTPUT_ERROR_SERVING_FAILURE_FILE_PATH, 'w') as s:
            s.write('Exception during inference: ' + str(err) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during inference: ' +
              str(err) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.

        return flask.Response(
            response="Error generating predictions. Check failure file.",
            status=400, mimetype="text/plain"
        )
