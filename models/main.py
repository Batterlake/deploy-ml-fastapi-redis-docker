import json
import os
import time

import im_compress as IC
import numpy as np
import psycopg2
import redis
import settings as S
from detection.yolo import Detector
from PIL import Image
from recognition.faces import get_face_embeddings
from recognition.plates import recognize_plates
from sklearn.metrics.pairwise import euclidean_distances

db = redis.StrictRedis(host=S.REDIS_HOST)
pgdb = psycopg2.connect(dbname="tdbm", user="service", password="123456", host="ml.n19")


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


def area(frame: np.ndarray):
    return frame.shape[0] * frame.shape[1]


def process_frontend_request(detector, queue_key):
    imageIDs, batch = collect_images(queue_key, 1)
    # Check to see if we need to process the batch
    if len(imageIDs) > 0:
        # Classify the batch
        pred = detector.predict(batch[0])
        faces = pred["faces"]
        face_areas = list(map(area, faces))
        face = faces[np.argmax(face_areas)]
        # plates = pred["plates"]

        embedding = get_face_embeddings([face])[0]

        db.set(imageIDs[0], json.dumps({"embedding": embedding.tolist()}))
        return True
    return False


def get_file_from_fs(data_folder: str):
    filename = np.random.choice(os.listdir(data_folder))
    print(f"FILENAME: {filename}")
    return Image.open(f"{data_folder}/{filename}").convert("RGB")


# def commit_results(results):
def check_license(text):
    cursor = pgdb.cursor()
    cursor.execute(f"select (uid, name, plate) from plates where (plate='{text}')")
    result = cursor.fetchone()
    if result is None:
        return None
    return result[0][0]


def query_closest(embedding: np.ndarray):
    # print(embedding)
    cursor = pgdb.cursor()
    cursor.execute(
        f"SELECT (uid, name, embedding) FROM faces ORDER BY embedding <-> '{embedding.tolist()}' LIMIT 1;"
    )
    result = cursor.fetchone()

    if result is None:
        return False, None

    rr = result[0].replace("(", "").replace(")", "")
    uid, name = rr.split(",")[:2]
    vec = eval(rr.split('"')[1])
    dst = euclidean_distances([embedding], [vec])[0][0]
    if dst > 2:
        return False, vec
    return True, uid


def commit_known_license(image: np.ndarray, license_uid: str, plate_text: str):
    image_bytes = IC.optimize_to_send(image, False)

    cursor = pgdb.cursor()
    cursor.execute(
        f"insert into events (user_id, plate, img_region) values ({license_uid}, {plate_text}, %s)",
        (image_bytes,),
    )
    pgdb.commit()


def commit_unknown_license(image: np.ndarray, plate_text: str):
    image_bytes = IC.optimize_to_send(image, False)

    cursor = pgdb.cursor()
    cursor.execute(
        f"insert into events (plate, img_region) values ('{plate_text}', %s)",
        (image_bytes,),
    )
    pgdb.commit()


def commit_known_face(image: np.ndarray, face_uid: str, face_vector: np.ndarray):
    image_bytes = IC.optimize_to_send(image, False)

    cursor = pgdb.cursor()
    cursor.execute(
        f"insert into events (user_id, embedding, img_region) values ({face_uid}, '{face_vector.tolist()}', %s)",
        (image_bytes,),
    )
    pgdb.commit()


def commit_unknown_face(image: np.ndarray, face_vector: np.ndarray):
    image_bytes = IC.optimize_to_send(image, False)

    cursor = pgdb.cursor()
    cursor.execute(
        f"insert into events (embedding, img_region) values ('{face_vector.tolist()}', %s)",
        (image_bytes,),
    )
    pgdb.commit()


def process_fs(
    detector,
    data_folder: str,
):
    print("******")
    image = get_file_from_fs(data_folder)
    result = detector.predict(image)
    faces = result["faces"]
    plates = result["plates"]

    if len(plates) > 0:
        # print("Has plates")
        plate_texts = recognize_plates(plates)
        license = None
        if len(plate_texts):

            license = check_license(plate_texts[0])
            print(plate_texts[0], license)
            if license is not None:
                print("known license")
            #     commit_known_license(plates[0], license, plate_texts[0])
            else:
                pass
                # print("UNKNOWN LICENSE")
            #     commit_unknown_license(plates[0], plate_texts[0])
        else:
            # print("OCR len 0")
            pass
    else:
        # print("NO PLATES")
        pass

    if len(faces):
        # print("has face")
        embedding = np.array(get_face_embeddings([faces[0]])[0])

        found_closest, obj_ = query_closest(embedding)
        if found_closest and obj_ is not None:
            print("found closest")
            # commit_known_face(faces[0], obj_, embedding)  # ?!?!?!
        else:
            pass
            # print("NO CLOSEST")
            # commit_unknown_face(faces[0], embedding)

    else:
        pass
        # print("NO FACES")


def embed_process(detector, queue_key: str, data_folder: str):
    # Continually poll for new images to classify
    while True:
        # Pop off multiple images from Redis queue atomically
        process_frontend_request(detector, queue_key)
        process_fs(detector, data_folder)
        timeout()


if __name__ == "__main__":
    d = Detector()
    embed_process(d, S.IMAGE_QUEUE, S.DATA_FOLDER)
