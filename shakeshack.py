from darkflow.net.build import TFNet
import numpy as np
import cv2

# model settings
options = {"model": "cfg/tiny-yolo.cfg", "load": "bin/tiny-yolo.weights", "threshold": 0.1}
tfnet = TFNet(options)

url = "https://cdn.shakeshack.com/camera.jpg"

while(True):
    cap = cv2.VideoCapture(url)
    ret, frame = cap.read()
    count = 0

    if ret:
	    # resize frame
        frame = cv2.resize(frame,(640,480))

        # get result from darkflow
        results = tfnet.return_predict(frame)
        for result in results:
            # display data if confidence is greater than 0.001
            confidence = result["confidence"]
            if confidence > 0.001:
                # top left cordinate
                tx = result["topleft"]["x"]
                ty = result["topleft"]["y"]
                # bottom right cordinate
                bx = result["bottomright"]["x"]
                by = result["bottomright"]["y"]
                # get label ex: person
                label = result["label"]
                if label == "person":
                    count = count + 1
                # draw rectangle
                cv2.rectangle(frame, (tx,ty), (bx,by), (0,0,255), thickness=2)
                font = cv2.FONT_HERSHEY_PLAIN
                text = label
                # display text above a rectangle
                cv2.putText(frame,text,(tx,ty-10),font, 2, (255, 255, 255), 2, cv2.LINE_AA)
				# display the number of people 
                cv2.putText(frame,"people: " + str(count), (20,30), font, 2, (255,0,0), 2, cv2.LINE_AA)

                # cv2.imshow("video", frame)
                # cv2.imshow('https://cdn.shakeshack.com/camera.jpg', frame)
                # overwrite frame on img.jpg
                cv2.imwrite('img.jpg', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
