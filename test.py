import cv2
import random
import time

bikes_classifier=cv2.CascadeClassifier('D:/Temp/Vehicle-And-Pedestrian-Detection-Using-Haar-Cascades-master/Vehicle-And-Pedestrian-Detection-Using-Haar-Cascades-master/Main Project/Main Project/Car Detection/cars.xml')
camera=cv2.VideoCapture(0)
offset = 150

while(True):
    ret,img=camera.read()
    
    height,width=img.shape[0:2]
    
    #img[0:70,0:width]=[0,0,255]
    #cv2.putText(img,'MOTOR BIKE COUNT:',(10,50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2)
    
    cv2.line(img,(width-offset,0),(width-offset,height),(0,255,255),2)
    cv2.line(img,(offset,0),(offset,height),(0,255,255),2)
    
    blur=cv2.blur(img,(3,3))
    gray=cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    
    bikes = bikes_classifier.detectMultiScale(gray)
    
    for (x,y,w,h) in bikes:
        
        bikeCy=int(x+w/2)
        linCx_max=width-offset
        linCx_min=offset
        
        if(bikeCy>linCx_min and bikeCy<linCx_max):
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(img,"Car",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img,"Car",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

        #cv2.putText(img,str(count),(500,50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2)
        
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    if key==27:
        break

cv2.destroyAllWindows()
camera.release()