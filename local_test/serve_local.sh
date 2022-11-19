#!/bin/sh

image=$1

docker run -v $(pwd)/ml_vol:/opt/ml_vol \
-e MODEL_SERVER_WORKERS=2 \
-e MODEL_SERVER_TIMEOUT=360 \
-p 8080:8080 --rm ${image} serve