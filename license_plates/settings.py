import os

REDIS_HOST: str = os.environ.get("REDIS_HOST")
IMAGE_QUEUE: str = os.environ.get("IMAGE_QUEUE")

SERVER_SLEEP = float(os.environ.get("SERVER_SLEEP"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
