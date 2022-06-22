import cv2
import time
import numpy as np
"""import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
GPIO.setup(7, GPIO.OUT)"""

cars_classifier=cv2.CascadeClassifier('D:/Temp/Car Detection/accuratelyModel.xml')
camera=cv2.VideoCapture(0)

f_width = 640
f_height = 480

offset = 200
start_line=offset-50
end_line=f_width-offset-50
velocity = 0
road_length=11
start_time = 0
end_time = 0

color = (0,255,0) #green
alert_color = (0,255,0)
flag = 0
max_time = 0

while(True):
    time.sleep(.05)

    ret,img=camera.read()
    
    img = cv2.resize(img, (f_width, f_height))
    height,width=img.shape[0:2]
    
    cv2.line(img,(start_line,0),(start_line,height),(0,0,255),2)
    cv2.line(img,(end_line,0),(end_line,height),(255,0,0),2)
    
    blur=cv2.blur(img,(3,3))
    gray=cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    
    cars = cars_classifier.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=10)
    
    for (x,y,w,h) in cars:

        carCx=int(x+w/2)
        if (carCx < start_line or carCx > end_line):
            color = (0,255,0)
        else:
            color = (0,0,255)
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        #cv2.putText(img,velocity,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)


        if (carCx > start_line and carCx < end_line and flag == 0):
            start_time = time.time()
            flag = 1
            max_time=0
            print("startTime :" + str(start_time))        
        if (carCx > end_line and flag == 1):
            end_time = time.time()
            flag = 2
            print("endTime :" + str(end_time), end="\n")
        if (flag == 2):
            pass_time = end_time - start_time
            if max_time < pass_time:
                max_time = pass_time     
            print(str(max_time) + "s")
            velocity = road_length / max_time #for cm/s
            velocity = round(velocity,2)
            #velocity=velocity*0.036 #for km/h
            print(velocity)
            flag = 0
            if (velocity > 30):
                #GPIO.output(7, True)
                alert_color = (0,0,255)
            else:
                #GPIO.output(7, False)
                alert_color = (0,255,0)

    cv2.putText(img,"Van toc cua xe: "+str(velocity)+"cm/s",(30,40),cv2.FONT_HERSHEY_COMPLEX,1,alert_color,2)
        
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(15)
    if key == 27:
        break

cv2.destroyAllWindows()
camera.release()