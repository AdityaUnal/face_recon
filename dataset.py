import cv2

face_classifier = cv2.CascadeClassifier(
    'C:/opencv-master/samples/winrt_universal/VideoCaptureXAML/video_capture_xaml/video_capture_xaml.Windows/Assets/haarcascade_frontalface_alt.xml')


def face_extractor(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converts the image to grayscale
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)  # finding faces

    if len(faces) == 0:
        return None
    else:
        crop_face = faces
        for (x, y, w, h) in faces:
            crop_face = img[y:y + h, x:x + w]
        return cv2.resize(crop_face, (200, 200))


cap = cv2.VideoCapture(0)

count = 0

while count != 100:
    ret, frame = cap.read()
    if ret is False:
        print("Error: Unable to capture frame.")
        break

    photo = face_extractor(frame)
    count = count + 1
    if photo is not None:
        face = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)

        path = 'D:/face recognition/faces/' + str(count) + '.jpg'
        cv2.imwrite(path, face)

        cv2.putText(face, str(count), (50, 50), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
        cv2.imshow('Cropped Face', face)

    else:
        print("No face found")
        pass

    count = count + 1
cap.release()
cv2.destroyAllWindows()

print("Collection successfully stored")
