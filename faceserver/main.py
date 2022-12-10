import json
import time
from argparse import ArgumentParser

import model as M
import redis
import settings as S
import utils as U

# Connect to Redis server
db = redis.StrictRedis(host=S.REDIS_HOST)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-s", dest="service", help="type of service", type=int, default=0
    )
    return parser.parse_args()


def get_batch_from_queue(queue_key: str, batch_size: int):
    with db.pipeline() as pipe:
        pipe.lrange(queue_key, 0, batch_size - 1)
        pipe.ltrim(queue_key, batch_size, -1)
        queue, _ = pipe.execute()
    return queue


def timeout():
    time.sleep(S.SERVER_SLEEP)


def collect_images(queue_key, batch_size):
    imageIDs = []
    batch = []
    queue = get_batch_from_queue(queue_key, batch_size)
    for q in queue:
        # Deserialize the object and obtain the input image
        q = json.loads(q.decode("utf-8"))
        batch.append(U.decode_image(q["image"], q["shape"]))

        # Update the list of image IDs
        imageIDs.append(q["id"])
    return imageIDs, batch


def embed_process(queue_key: str, batch_size: int, recognizer_backend: str):
    # Continually poll for new images to classify
    while True:
        # Pop off multiple images from Redis queue atomically
        imageIDs, batch = collect_images(queue_key, batch_size)
        # Check to see if we need to process the batch
        if len(imageIDs) > 0:
            # Classify the batch
            print("* Batch size: {}".format(len(batch)))
            embeddings = M.recognize_faces(batch, recognizer_backend)

            # Loop over the image IDs and their corresponding set of results from our model
            for uid, emb in zip(imageIDs, embeddings):
                # Initialize the list of output predictions
                # Store the output predictions in the database, using image ID as the key so we can fetch the results
                db.set(uid, json.dumps({"embedding": emb.tolist()}))

        # Sleep for a small amount
        timeout()


def detect_embed_process(
    queue_key: str, detector_backend: str, recognizer_backend: str
):
    # Continually poll for new images to classify
    while True:
        # Pop off multiple images from Redis queue atomically
        imageIDs, batch = collect_images(queue_key, 1)
        # Check to see if we need to process the batch
        if len(imageIDs) > 0:
            uid = imageIDs[0]
            image = batch[0]
            faces, locations, embeddings = M.detect_embedd(
                image, detector_backend, recognizer_backend
            )
            embeddings = embeddings if embeddings is not None else []
            locations = locations if locations is not None else []
            output = []
            # Loop over the image IDs and their corresponding set of results from our model
            for f, l, emb in zip(faces, locations, embeddings):
                # Initialize the list of output predictions
                # Store the output predictions in the database, using image ID as the key so we can fetch the results
                output.append(
                    {
                        "face": U.encode_image(f),
                        "location": l,
                        "embedding": emb.tolist(),
                        "shape": f.shape,
                    }
                )
            db.set(
                uid,
                json.dumps(output),
            )

        # Sleep for a small amount
        timeout()


if __name__ == "__main__":
    args = parse_args()
    if args.service == 0:
        detect_embed_process(
            S.DETECT_IMAGE_QUEUE, S.DETECTOR_BACKEND, S.RECOGNIZER_BACKEND
        )
    else:
        embed_process(S.EMBED_IMAGE_QUEUE, S.BATCH_SIZE, S.RECOGNIZER_BACKEND)
