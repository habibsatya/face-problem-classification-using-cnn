import cv2
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
base_dir = 'image-test/t15.jpg'

def createBox(img, points):
    bbox = cv2.boundingRect(points)
    x, y, w, h = bbox
    imgCrop = img[y:y+h, x:x+w]
    imgCrop = cv2.resize(imgCrop, (200, 200))
    return imgCrop

img = cv2.imread(base_dir)
img_original = img.copy
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector(img_gray)

for face in faces:
    x1, y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()
    # img_original = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    landmarks = predictor(img_gray, face)
    myPoints = []
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        myPoints.append([x, y])
    
    myPoints = np.array(myPoints)
    img_cheek_right = createBox(img, myPoints[1:7])
    img_cheek_left = createBox(img, myPoints[10:16])
    img_cheek = np.concatenate((img_cheek_right, img_cheek_left), axis=1)
# cv2.imwrite('test.jpg', img_nose)
# cv2.imshow('right side cropped', img_cheek_right)
# cv2.imshow('left side cropped', img_cheek_left)
cv2.imshow('cheek cropped', img_cheek)
cv2.waitKey(0)
