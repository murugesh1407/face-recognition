import cv2
import numpy as np

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
eyeDetect =cv2.CascadeClassifier('haarcascade_eye.xml');
cam=cv2.VideoCapture(0);
rec=cv2.face.LBPHFaceRecognizer_create();
rec.read("recognizer\\trainingData.yml")
Id=0
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1 #2
fontcolor = (255, 255, 255)
size=3 #3
while(True):
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.rectangle(img,(x-20,y-20),(x+w+20,y+h+20),(0,255,0),4)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]
        eyes=eyeDetect.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        Id,conf=rec.predict(gray[y:y+h,x:x+w])
        if (Id==1):
            Id="Murugesh"
            f=open("id1.txt",'w')
            f.write("Name:Murugesh\nage:20\nReg:035\nDept:CSE")
        elif (Id==2):
            Id="Vishal"
            f=open("id2.txt",'w')
            f.write("Name:Test\nage:23\nReg:89\nDept:Ece")
            
        cv2.rectangle(img,(x-22,y-90),(x+w+22,y-22),(0,255,0),-1)
        cv2.putText(img,str(Id),(x,y-40), fontface, fontscale, fontcolor,size);
        #y+h
    cv2.imshow("Face",img);
    if(cv2.waitKey(1)==ord('q')):
        break;
cam.release()
cv2.destroyAllWindows()
