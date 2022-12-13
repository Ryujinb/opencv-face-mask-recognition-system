import tensorflow as tf
import numpy as np

import cv2

import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

from uuid import uuid4
import time
import os

# Firebase storage 인증 및 앱 초기화
cred = credentials.Certificate('dbkey.json')  # 경로를 파이참 프로젝트 외부로 지정하고 인증키를 절대로 업로드 하지 마세요.
firebase_admin.initialize_app(cred, {
    'storageBucket': 'opencv-mask.appspot.com'
})
bucket = storage.bucket()  # 기본 버킷 사용


# firestorage에 파일 업로드를 하기 위한 함수
def fileUpload(file):
    blob = bucket.blob('photo/' + file)
    # new token and metadata 설정
    new_token = uuid4()
    metadata = {"firebaseStorageDownloadTokens": new_token}  # access token이 필요하다.
    blob.metadata = metadata
    # upload file
    blob.upload_from_filename(filename='./temp_photo/' + str(now) + ".jpg")


# Open the cam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)

# Load the saved model
model = tf.keras.models.load_model("keras_model.h5", compile=False)
# Define the input size of the model
input_size = (224, 224)

while cap.isOpened():
    # Reading frames from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for the model
    model_frame = cv2.resize(frame, input_size, frame)
    # Expand Dimension (224, 224, 3) -> (1, 224, 224, 3) and Normalize the data
    model_frame = np.expand_dims(model_frame, axis=0) / 255.0

    # Predict
    is_mask_prob = model.predict(model_frame)[0]
    is_mask = np.argmax(is_mask_prob)

    # Add Information on screen
    if is_mask == 0:
        msg_mask = "mask"
        now = time.strftime('%Y-%m-%d %H %M %S', time.localtime(time.time()))
        cv2.imwrite('./temp_photo/' + str(now) + ".jpg", frame)
        fileUpload(now)
        time.sleep(2)
        os.remove('./temp_photo/' + str(now) + ".jpg")
    elif is_mask == 1:
        msg_mask = "no mask"
    else:
        msg_mask = "nothing"

    msg_mask += " ({:.1f})%".format(is_mask_prob[is_mask] * 100)
    cv2.putText(frame, msg_mask, (10, 30), cv2.FONT_ITALIC, 1, (51, 102, 0), thickness=2)

    # Show the result and frame
    cv2.imshow('face mask detection', frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
