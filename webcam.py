from darkflow.net.build import TFNet
import numpy as np
import cv2

# model settings
options = {"model": "cfg/tiny-yolo.cfg", "load": "bin/tiny-yolo.weights", "threshold": 0.1}
tfnet = TFNet(options)

# use built-in webcam
cap = cv2.VideoCapture(0)

# frame size
frame_width = 640
frame_height = 480
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while(cap.isOpened()):
    
    ret, frame = cap.read()
    frame = cv2.resize(frame,(640,480))
    results = tfnet.return_predict(frame)
    for result in results:
        # display data if confidence is greater than 0.5
        confidence = result["confidence"]
        if confidence > 0.5:
            # top left cordinate
            tx = result["topleft"]["x"]
            ty = result["topleft"]["y"]
            # bottom right cordinate
            bx = result["bottomright"]["x"]
            by = result["bottomright"]["y"]
            # get label ex: person
            label = result["label"]
            
            # draw rectangle
            cv2.rectangle(frame, (tx,ty), (bx,by), (0,0,255), thickness=2)
            font = cv2.FONT_HERSHEY_PLAIN
            text = label
            cv2.putText(frame,text,(tx,ty-10),font, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("video", frame)
            
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
