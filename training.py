import os
import cv2
import numpy as np

data_path = 'D:/face recognition/faces/'
files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

Training_Data = []
labels = []

# for i, files in enumerate(files):
#     image_path = data_path + files[i]
#     images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if images is not None:
#         Training_Data.append(np.asarray(images, dtype=np.uint8))
#     lables.append(i)

for i, file_name in enumerate(files):
    image_path = data_path + files[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    labels.append(i)

labels = np.asarray(labels, dtype=np.int32)
model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data), np.asarray(labels))

print("Training complete")

