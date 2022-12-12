import json
import time
import uuid

import redis
import settings as S
from fastapi import FastAPI, File, HTTPException
from starlette.requests import Request

app = FastAPI()
db = redis.StrictRedis(host=S.REDIS_HOST)


@app.get("/")
def health_check():
    return "I'm alive!"


def process_image(queue_key: str, img_base64: str):
    data = {"success": False}

    # Generate an ID for the classification then add the classification ID + image to the queue
    k = str(uuid.uuid4())
    d = {"id": k, "image": img_base64}
    db.rpush(queue_key, json.dumps(d))

    # Keep looping for CLIENT_MAX_TRIES times
    num_tries = 0
    while num_tries < S.CLIENT_MAX_TRIES:
        num_tries += 1

        # Attempt to grab the output predictions
        output = db.get(k)

        # Check to see if our model has classified the input image
        if output is not None:
            # Add the output predictions to our data dictionary so we can return it to the client
            output = output.decode("utf-8")
            data["predictions"] = json.loads(output)

            # Delete the result from the database and break from the polling loop
            db.delete(k)
            break

        # Sleep for a small amount to give the model a chance to classify the input image
        time.sleep(S.CLIENT_SLEEP)

        # Indicate that the request was a success
        data["success"] = True
    else:
        raise HTTPException(
            status_code=400,
            detail="Request failed after {} tries".format(S.CLIENT_MAX_TRIES),
        )

    # Return the data dictionary as a JSON response
    return data


# @app.post("/detect_face")
# def detect(
#     request: Request,
#     img_base64: str = File(...),
# ):
#     return process_image(S.DETECT_IMAGE_QUEUE, img_base64)


# @app.post("/embed_face")
# def embed(
#     request: Request,
#     img_base64: str = File(...),
# ):
#     return process_image(S.EMBED_IMAGE_QUEUE, img_base64)


@app.post("/embed_face_2")
def embed(
    request: Request,
    img_base64: str = File(...),
):
    return process_image(S.IMAGE_QUEUE, img_base64)


# @app.post("/license_plate")
# def plate(
#     request: Request,
#     img_base64: str = File(...),
# ):

#     return process_image(S.IMAGE_QUEUE, img_base64)
