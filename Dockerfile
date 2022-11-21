ARG MODEL_FOLDER_PATH="/models"


FROM python:3.8.0-slim as init
ARG MODEL_FOLDER_PATH

COPY ./requirements.txt .
RUN pip3 install -r requirements.txt 

ENV PATH="/opt/app:${PATH}"
ENV PRETRAINED_MODEL_FOLDER_PATH=${MODEL_FOLDER_PATH}

COPY app/init /opt/app/init
COPY app/algorithm/misc /opt/app/algorithm/misc
COPY app/algorithm/initializer /opt/app/algorithm/initializer
WORKDIR /opt/app

RUN chmod +x init
RUN init


FROM python:3.8.0-slim as builder
ARG MODEL_FOLDER_PATH

RUN apt-get -y update && apt-get install -y --no-install-recommends \
    wget \
    nginx \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*


COPY ./requirements.txt .
RUN pip3 install -r requirements.txt 

COPY app /opt/app
WORKDIR /opt/app

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/app:${PATH}"
ENV PRETRAINED_MODEL_FOLDER_PATH=${MODEL_FOLDER_PATH}

COPY --from=init ${MODEL_FOLDER_PATH} ${MODEL_FOLDER_PATH}

RUN chmod +x train &&\
    chmod +x test &&\
    chmod +x tune &&\
    chmod +x serve 