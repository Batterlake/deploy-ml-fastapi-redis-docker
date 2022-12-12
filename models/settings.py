import os

REDIS_HOST: str = os.environ.get("REDIS_HOST")
IMAGE_QUEUE: str = os.environ.get("EMBED_IMAGE_QUEUE")
SERVER_SLEEP = float(os.environ.get("SERVER_SLEEP"))
DATA_FOLDER: str = os.environ.get("DATA_FOLDER")
