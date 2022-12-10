import os

REDIS_HOST: str = os.environ.get("REDIS_HOST")
DETECT_IMAGE_QUEUE: str = os.environ.get("DETECT_IMAGE_QUEUE")
EMBED_IMAGE_QUEUE: str = os.environ.get("EMBED_IMAGE_QUEUE")

SERVER_SLEEP = float(os.environ.get("SERVER_SLEEP"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))

DETECTOR_BACKEND: str = os.environ.get("DETECTOR_BACKEND")
RECOGNIZER_BACKEND: str = os.environ.get("RECOGNIZER_BACKEND")
