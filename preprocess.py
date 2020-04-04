
from parameters import *
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from os.path import join

cascade_classifier = cv2.CascadeClassifier(CASC_PATH)


def format_image(image, flag=0):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    if flag == 0:
        gray_border = np.zeros((150, 150), np.uint8)
        gray_border[:, :] = 200
        gray_border[
            int((150 / 2) - (SIZE_FACE / 2)): int((150 / 2) + (SIZE_FACE / 2)),
            int((150 / 2) - (SIZE_FACE / 2)): int((150 / 2) + (SIZE_FACE / 2))
        ] = image
        image = gray_border
    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor=SCALEFACTOR,
        minNeighbors=5
    )

   
    if not len(faces) > 0:
        return None
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face

    face = max_area_face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
  
    try:
        image = cv2.resize(image, (SIZE_FACE, SIZE_FACE),
                           interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("[+] Problem during resize")
        return None
    return image


def emotion_to_vec(x):
    d = np.zeros(len(EMOTIONS))
    d[x] = 1.0
    return d


def flip_image(image):
    return cv2.flip(image, 1)


def data_to_image(data, i):
    data_image = np.fromstring(
        str(data), dtype=np.uint8, sep=' ').reshape((SIZE_FACE, SIZE_FACE))
    data_image = Image.fromarray(data_image).convert('RGB')
    data_image = np.array(data_image)[:, :, ::-1].copy()

    data_image = format_image(data_image)
    return data_image


def get_fer(csv_path):
    data = pd.read_csv(csv_path)
    labels = []
    images = []
    count = 0
    total = data.shape[0]

    for index, row in data.iterrows():
        emotion = emotion_to_vec(row['emotion'])
        image = data_to_image(row['pixels'], index)

        if image is not None:
            labels.append(emotion)
            images.append(image)

            count += 1
        print("Progress: {}/{} {:.2f}%".format(index, total, index * 100.0 / total))

    print(index) 
    print("Total: " + str(len(images)))
    np.save(join(SAVE_DIRECTORY, SAVE_DATASET_IMAGES_FILENAME), images)
    np.save(join(SAVE_DIRECTORY, SAVE_DATASET_LABELS_FILENAME), labels)


if __name__ == '__main__':
    get_fer(join(SAVE_DIRECTORY, DATASET_CSV_FILENAME))
    pass
