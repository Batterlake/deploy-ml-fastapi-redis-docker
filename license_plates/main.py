import json
import time

import im_compress as IC
import redis
import settings as S
from ocr.langs import Eu, LangWrapper

# Connect to Redis server
db = redis.StrictRedis(host=S.REDIS_HOST)
model = LangWrapper(Eu())


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
        batch.append(IC.image_from_base64(q["image"]))
        # batch.append(U.decode_image(q["image"], q["shape"]))

        # Update the list of image IDs
        imageIDs.append(q["id"])
    return imageIDs, batch


def license_plate_process(queue_key: str, batch_size: int):
    # Continually poll for new images to classify
    while True:
        # Pop off multiple images from Redis queue atomically
        imageIDs, batch = collect_images(queue_key, batch_size)
        # Check to see if we need to process the batch
        if len(imageIDs) > 0:
            # Classify the batch
            print("* Batch size: {}".format(len(batch)))
            license_plates = model.predict_images(batch)

            # Loop over the image IDs and their corresponding set of results from our model
            for uid, lp in zip(imageIDs, license_plates):
                # Initialize the list of output predictions
                # Store the output predictions in the database, using image ID as the key so we can fetch the results
                db.set(uid, json.dumps({"license_plate": lp}))

        # Sleep for a small amount
        timeout()


if __name__ == "__main__":
    license_plate_process(
        S.IMAGE_QUEUE,
        S.BATCH_SIZE,
    )
