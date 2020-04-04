
import cv2
from parameters import *
from train import EmotionRecognition
import numpy as np

cascade_classifier = cv2.CascadeClassifier(CASC_PATH)

network = EmotionRecognition()
network.build_network()


def face_predict(face, gray):
    gray = gray[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
    try:
        gray = cv2.resize(gray, (SIZE_FACE, SIZE_FACE),
                           interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("[+] Problem during resize")
        return None

    result = network.predict(gray)

    return result


if __name__ == '__main__':
    pic_path = 'testgallery/1.jpg'

    image = cv2.imread(pic_path)

    if len(image.shape) > 2 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.imencode(image, cv2.IMREAD_GRAYSCALE)

    faces = cascade_classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )
    for face in faces:
        (x, y, w, h) = face
        cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)
        print(x, y, w, h)
        result = face_predict(face, gray)
        text = EMOTIONS[np.argmax(result[0])]
        print(result, text)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    cv2.imwrite('detected_faces.png', image)
