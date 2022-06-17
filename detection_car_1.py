
import cv2
cascade_src = 'D:/Temp/Vehicle-And-Pedestrian-Detection-Using-Haar-Cascades-master/Vehicle-And-Pedestrian-Detection-Using-Haar-Cascades-master/Main Project/Main Project/Car Detection/cars.xml'

video_src = 'D:/Temp/Vehicle-And-Pedestrian-Detection-Using-Haar-Cascades-master/Vehicle-And-Pedestrian-Detection-Using-Haar-Cascades-master/Main Project/Main Project/Car Detection/video.avi'

cap = cv2.VideoCapture(0)

car_cascade = cv2.CascadeClassifier(cascade_src)


while True:
    ret, img = cap.read()
   
    if (type(img) == type(None)):
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 2)


    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
    
    cv2.imshow('video', img)
   
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
