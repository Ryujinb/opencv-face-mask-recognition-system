import tensorflow as tf
import numpy as np

import cv2

import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

from uuid import uuid4
import time
import os

import serial

checked = False
frame_count = 0

# 파이참과 아두이노 통신을 위한 보드 연결
ad = serial.Serial(
    port='COM3',
    baudrate=9600,
)


# Firebase storage 인증 및 앱 초기화
cred = credentials.Certificate('dbkey.json')  # 경로를 파이참 프로젝트 외부로 지정하고 인증키를 절대로 업로드 하지 마세요.
firebase_admin.initialize_app(cred, {
    'storageBucket': 'opencv-mask.appspot.com'
})

# 기본 버킷 사용
bucket = storage.bucket()


# firestorage에 파일 업로드를 하기 위한 함수
def fileUpload(file):
    #저장 위치 지정
    blob = bucket.blob('photo/' + file)
    # new token and metadata 설정
    new_token = uuid4()
    metadata = {"firebaseStorageDownloadTokens": new_token}
    blob.metadata = metadata
    # 로컬로부터 파일 업로드
    blob.upload_from_filename(filename='./temp_photo/' + str(now) + ".jpg")


# 캠 열기
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)

# 저장된 모델 불러오기
model = tf.keras.models.load_model("keras_model.h5", compile=False)
# 모델 input 사이즈 정의
input_size = (224, 224)
msg_mask = ""

while cap.isOpened():
    frame_count += 1
    # 캠으로부터 프레임 읽어오기
    ret, frame = cap.read()
    if not ret:
        break

    # 모델 위해 프레임 크기 재지정
    model_frame = cv2.resize(frame, input_size, frame)
    # 데이터 일반화
    model_frame = np.expand_dims(model_frame, axis=0) / 255.0

    # 예측
    is_mask_prob = model.predict(model_frame)[0]
    is_mask = np.argmax(is_mask_prob)

    # 각 상황에 맞는 행동을 위한 조건문
    # 50 프레임 마다 한번씩 실행
    if frame_count % 50 == 0:
        # 마스크를 쓰고 있으면
        if is_mask == 0:
            msg_mask = "Mask detected"
            # checked가 false일 경우만 업로드 및 데이터 전송을 한다.
            if not checked:
                now = time.strftime('%Y-%m-%d %H %M %S', time.localtime(time.time()))
                # 현재날짜,시간을 파일 이름으로 하여 로컬에 캡처저장
                cv2.imwrite('./temp_photo/' + str(now) + ".jpg", frame)
                #업로드 함수를 통한 업로드
                fileUpload(now)
                checked = True
                # 로컬에 저장된 사진은 삭제
                os.remove('./temp_photo/' + str(now) + ".jpg")
                # 아두이노에 1을 전송
                mask = '1'
                mask = mask.encode('utf-8')
                ad.write(mask)

        # 마스크를 안쓰고 있으면
        elif is_mask == 1:
            msg_mask = "No mask"
            # 아두이노에 2를 전송
            mask = '2'
            mask = mask.encode('utf-8')
            ad.write(mask)

        # 빈 벽일시
        else:
            msg_mask = "Nothing"
            # 빈 벽이 나오면 checked를 false로 바꾼다.
            checked=False

        msg_mask += " ({:.1f})%".format(is_mask_prob[is_mask] * 100)

    # 메시지 출력
    cv2.putText(frame, msg_mask, (10, 30), cv2.FONT_ITALIC, 1, (51, 102, 0), thickness=2)

    cv2.imshow('face mask detection', frame)

    # q 눌러 캠 끄기
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break