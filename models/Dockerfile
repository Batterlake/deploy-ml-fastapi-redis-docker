FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
ENV DEBIAN_FRONTEND noninteractive
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update

# For pip modules
RUN apt-get install -y git
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install ffmpeg libsm6 libxext6  -y
# turbojpeg
RUN apt-get install -y libturbojpeg

RUN pip3 install opencv-python
RUN pip3 install yolov5
RUN pip3 install deepface
RUN pip3 install torch
RUN pip3 install torchvision
RUN pip3 install git+https://github.com/ria-com/modelhub-client.git

RUN pip3 install redis
RUN pip3 install deepface
RUN pip3 install psycopg2-binary
RUN pip3 install scikit-learn

COPY . /app

CMD ["python", "/app/main.py"]
