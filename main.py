import cv2
import time
import numpy as np

cars_classifier=cv2.CascadeClassifier('D:/Temp/Car Detection/accuratelyModel.xml')
camera=cv2.VideoCapture(0)

f_width = 1280
f_height = 720

offset2 = 150
my_min = offset2
my_max = f_width - offset2
offset = 400
velocity = 0
road_length=15
start_time = 0
end_time = 0

color = (0,255,0) #green
flag = 0
max_time = 0

while(True):
    time.sleep(.05)

    ret,img=camera.read()
    
    img = cv2.resize(img, (f_width, f_height))
    height,width=img.shape[0:2]
    
    cv2.line(img,(width-offset,0),(width-offset,height),(0,255,255),2)
    cv2.line(img,(offset,0),(offset,height),(255,0,0),2)
    
    blur=cv2.blur(img,(3,3))
    gray=cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    
    cars = cars_classifier.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=10)
    
    for (x,y,w,h) in cars:

        carCx=int(x+w/2)
        linCx_max=width-offset
        linCx_min=offset

        if (carCx>linCx_min and carCx<linCx_max):
            color = (0,0,255)
        if (carCx<linCx_min or carCx>linCx_max):
            color = (0,255,0)
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        #cv2.putText(img,velocity,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)


        if (carCx > linCx_min and carCx<linCx_max and flag == 0):
            start_time = time.time()
            flag = 1
            max_time=0
            print("startTime :" + str(start_time))        
        if (carCx > linCx_max and flag == 1):
            end_time = time.time()
            flag = 2
            print("endTime :" + str(end_time), end="\n")
        if (flag == 2):
            pass_time = end_time - start_time
            if max_time < pass_time:
                max_time = pass_time     
            print(str(max_time) + "s")
            velocity = road_length / max_time #for cm/s
            #velocity=velocity*0.036 #for km/h
            print(str(round(velocity,2)) + "cm/s")
            flag = 0
            break
        
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(15)
    if key==27:
        break

cv2.destroyAllWindows()
camera.release()