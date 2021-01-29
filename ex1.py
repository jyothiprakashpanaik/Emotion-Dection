from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('./Emotion_Detection.h5')
class_labels = ['Angry','Happy','Neutral','Sad','Suprise']
cap = cv2.VideoCapture(0)

while True:
	_,frame = cap.read()
	labels = []
	grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_classifier.detectMultiScale(grayImg,1.3,5)

	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),2)
		roi_grayImg = grayImg[y:y+h,x:x+w]
		roi_grayImg = cv2.resize(roi_grayImg, (48,48),interpolation=cv2.INTER_AREA)

		if np.sum([roi_grayImg])!=0:
			roi = roi_grayImg.astype('float')/255.0
			roi = img_to_array(roi)
			roi = np.expand_dims(roi, axis=0)

			pred = classifier.predict(roi)[0]
			print("\nprediction = ",pred)
			label = class_labels[pred.argmax()]
			print("\nprediction max = ",pred.argmax())
			print("\nlabel = ",label)
			label_position = (x,y)
			cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255))
		else:
			cv2.putText(frame, 'No Face Found', (20,60), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255))
			print("\n\n")
		cv2.imshow("Emotion Detector", frame)

		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			print("quit")
			break
cap.release()
cv2.destroyAllWindows()
