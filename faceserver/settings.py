import os

SERVER_SLEEP = float(os.environ.get("SERVER_SLEEP"))
REDIS_HOST: str = os.environ.get("REDIS_HOST")

IMAGE_QUEUE: str = os.environ.get("IMAGE_QUEUE")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))

DETECTOR_BACKEND: str = os.environ.get("DETECTOR_BACKEND")
RECOGNIZER_BACKEND: str = os.environ.get("RECOGNIZER_BACKEND")
