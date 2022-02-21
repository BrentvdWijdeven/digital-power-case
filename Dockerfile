# syntax=docker/dockerfile:1

FROM godatadriven/pyspark:latest

COPY requirements.txt requirements.txt

ENV KAGGLE_USERNAME = "brentvdwijdeven"
ENV KAGGLE_KEY = "a375d80cb669fffd5a1846b9fba5e1fd"

RUN pip3 install -r requirements.txt

COPY . .


RUN python3 main.py