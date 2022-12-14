#!/bin/sh



image=$1
if [ $# -eq 2 ]
then
    gpu="--gpus ${2}"
fi



mkdir -p ml_vol/inputs/data_config
mkdir -p ml_vol/inputs/data/testing/textClassificationBaseMainInput
mkdir -p ml_vol/inputs/data/testing/textClassificationBaseMainInput/keys

mkdir -p ml_vol/model/model_config
mkdir -p ml_vol/model/artifacts

mkdir -p ml_vol/hpt/results

mkdir -p ml_vol/outputs
mkdir -p ml_vol/outputs/testing_outputs
mkdir -p ml_vol/outputs/errors

chmod 777 ml_vol/model

cp examples/abalone_schema.json ml_vol/inputs/data_config
cp examples/abalone_test.csv ml_vol/inputs/data/testing/textClassificationBaseMainInput
cp examples/abalone_test_key.csv ml_vol/inputs/data/testing/textClassificationBaseMainInput/keys
cp examples/hyperparameters.json ml_vol/model/model_config

echo ${gpu}
docker run -v $(pwd)/ml_vol:/opt/ml_vol \
-e CHANNEL=testing \
-e EVALUATE_FILE='abalone_test_key.csv' \
${gpu} --rm ${image} test
