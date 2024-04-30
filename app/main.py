import requests
from PIL import Image
from io import BytesIO
import face_recognition
import numpy as np

def read_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    face_locations = face_recognition.face_locations(np.array(img))
    # Crop the detected face region from the original image
    top, right, bottom, left = face_locations[0]
    cropped_img = img.crop((left, top, right, bottom))
    
    return np.array(cropped_img)

def is_same_person(img1, img2, thread_hold = 0.6):
    img1_face_encoding = face_recognition.face_encodings(img1)[0]
    img2_face_encoding = face_recognition.face_encodings(img2)[0]
    return face_recognition.face_distance([img1_face_encoding], img2_face_encoding) < thread_hold

