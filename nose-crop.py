import cv2
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
base_dir = 'image-test/t3.jpg'

def createBox(img, points):
    bbox = cv2.boundingRect(points)
    x, y, w, h = bbox
    imgCrop = img[y-5:y+h+5, x-5:x+w+5]
    imgCrop = cv2.resize(imgCrop, (200, 200))
    return imgCrop

img = cv2.imread(base_dir)
img_original = img.copy
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector(img_gray)

for face in faces:
    x1, y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()
    img_original = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    landmarks = predictor(img_gray, face)
    myPoints = []
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        myPoints.append([x, y])
    
    myPoints = np.array(myPoints)
    img_nose = createBox(img, myPoints[27:36])

# cv2.imwrite('test.jpg', img_nose)
cv2.imshow('nose cropped', img_nose)
cv2.waitKey(0)
