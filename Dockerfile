FROM ubuntu:20.04

RUN apt update
RUN apt install -y python3 python3-pip

COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

RUN sudachipy link -t full

ENV APP_PATH=/home
WORKDIR ${APP_PATH}
ENV HOME=${APP_PATH}
ENV PYTHONPATH=${APP_PATH}