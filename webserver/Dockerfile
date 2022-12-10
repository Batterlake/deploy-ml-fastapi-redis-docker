FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY requirements.txt /app/

RUN pip install tensorflow-gpu
RUN pip install keras
RUN pip install Pillow
RUN pip install redis
RUN pip install python-multipart

COPY . /app
