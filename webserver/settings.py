import os

CLIENT_SLEEP = float(os.environ.get("CLIENT_SLEEP"))
CLIENT_MAX_TRIES = int(os.environ.get("CLIENT_MAX_TRIES"))
REDIS_HOST: str = os.environ.get("REDIS_HOST")
DETECT_IMAGE_QUEUE: str = os.environ.get("DETECT_IMAGE_QUEUE")
EMBED_IMAGE_QUEUE: str = os.environ.get("EMBED_IMAGE_QUEUE")