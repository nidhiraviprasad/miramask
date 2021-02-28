import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

#connects to video camera and checks whether detected faces are wearing masks

model = load_model('mask2.h5')

faceHaarCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
	ret, testImage = cap.read()
	if not ret:
		continue
	grayImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)

	facesDetected = faceHaarCascade.detectMultiScale(grayImage, 1.32, 5)

	for (x, y, w, h) in facesDetected:
		cv2.rectangle(testImage, (x, y), (x+w, y+h), (255,0,0), thickness=7)
		roiGray = grayImage[y:y+w, x:x+h]
		roiGray = cv2.resize(roiGray, (150,150))
		imagePix = image.img_to_array(roiGray)
		imagePix = np.expand_dims(imagePix, axis = 0)
		imagePix /= 255
		predictions = model.predict(imagePix)

		index = np.argmax(predictions[0])
		verdict = ('Mask!', 'No Mask!')
		predictedVerdict = verdict[index]

		if verdict[index] == 'Mask!':
			cv2.rectangle(testImage, (x, y), (x+w, y+h), (0,255,0), thickness=7)
			cv2.putText(testImage, verdict[index], (int(x), int(y)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0))
		else:
			cv2.rectangle(testImage, (x, y), (x+w, y+h), (0,0,255), thickness=7)
			cv2.putText(testImage, verdict[index], (int(x), int(y)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255))

	resized = cv2.resize(testImage, (1000,700))
	cv2.imshow('miramask', resized)

	if cv2.waitKey(10) == ord('q'):
		break


cap.release()
cv2.destroyAllWindows
