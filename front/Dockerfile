FROM python:3.9-slim-buster

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip3 install flask
RUN pip3 install psycopg2-binary
RUN pip3 install numpy

RUN pip3 install opencv-python
RUN pip3 install python-dotenv
RUN pip3 install requests

COPY . /app
WORKDIR /app

EXPOSE 5000