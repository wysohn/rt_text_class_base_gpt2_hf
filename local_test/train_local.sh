#!/bin/sh

image=$1

rm -rf ml_vol


mkdir -p ml_vol/inputs/data_config
mkdir -p ml_vol/inputs/data/training/textClassificationBaseMainInput

mkdir -p ml_vol/model/model_config
mkdir -p ml_vol/model/artifacts

mkdir -p ml_vol/hpt/results

mkdir -p ml_vol/outputs
mkdir -p ml_vol/outputs/errors

chmod 777 ml_vol/model

cp examples/abalone_schema.json ml_vol/inputs/data_config
cp examples/abalone_train.csv ml_vol/inputs/data/training/textClassificationBaseMainInput
cp examples/hyperparameters.json ml_vol/model/model_config

docker run -v $(pwd)/ml_vol:/opt/ml_vol --rm ${image} train
