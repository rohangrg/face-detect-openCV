import numpy as np 
import cv2


facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

sampleNum = 0

uid = input('enter user id')

cam = cv2.VideoCapture(0)

while(True):
	ret,img = cam.read()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = facedetect.detectMultiScale(gray,1.3,5)

	for(x,y,w,h) in faces:
		sampleNum+=1
		cv2.imwrite('dataset/' + str(uid) + '_' + str(sampleNum) + '.jpg',gray[y:y+h , x:x+h])
		cv2.rectangle(img , (x,y) , (x+w , y+h) , (0,255,0) , 2)
		cv2.waitKey(300)
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

	cv2.imshow("face",img)
	if(cv2.waitKey(1)==ord('q')):
		break
	if(sampleNum>50):
		break

cam.release()
cv2.destroyAllWindows()