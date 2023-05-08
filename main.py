import pathlib
import cv2


# Read the Image
imagePath = 'face.jpg'
img = cv2.imread(imagePath)


# Convert the Image to Grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_image.shape


# Load the Classifier Algorithm
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Perform the Face Detection
face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)

# Draw a Bouding Box
for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
    faces = img[y:y + h, x:x + w]
    cv2.imwrite('face-cropped.jpg', faces)

#Save Image
cv2.imwrite('face-detected.jpg', img)
# cv2.imshow('img', img)
cv2.waitKey()




