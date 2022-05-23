import cv2

def face_detection(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, 1.3, 4) ### The image, scale factor, min neghbor
    # itertion for multi fce detection
    for (x, y, w, h) in faces:
        cv2.rectangle(gray_img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return gray_img

