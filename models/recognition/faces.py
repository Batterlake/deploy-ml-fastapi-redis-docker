from typing import List

from deepface import DeepFace
import numpy as np

# options: ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "Dlib", "ArcFace"]
RECOGNIZER_BACKEND = 'Facenet'

def get_face_embeddings(
    images: List[np.ndarray], 
    backend: str=RECOGNIZER_BACKEND
) -> List[List[np.ndarray]]:
    embed = lambda face: DeepFace.represent(face, 
                                            model_name=backend, 
                                            enforce_detection=False, 
                                            detector_backend='skip',
                                            align=False) 
    return list(map(embed, images))
