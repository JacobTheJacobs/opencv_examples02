import numpy as np
import cv2
import argparse
import time
import imutils
from imutils.video import VideoStream

ap = argparse.ArgumentParser()

#construct argument parser and parse
ap.add_argument("-p", "--prototxt", required=True, help="deploy.prototxt.txt")
ap.add_argument("-m", "--model", required=True, help="res10_300x300)ssd)iter_140000.caffemodel")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help=0.5)

args = vars(ap.parse_args())

# load the model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# init video stream
print("[INFO] starting stream...")
vs = VideoStream(0).start()
time.sleep(2.0)

#loop over the frames
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    #grab he frame demensions and convert it blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    #pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    #loop over the detections
    for i in range(0, detections.shape[2]):
        #extract the confidence (i,e probability) assossiated with prediction
        confidence = detections[0, 0, i, 2]

#filter out weak detections by ensuring the confidence is greater then the minimum confidence
        if confidence < args["confidence"]:
            continue

#compute the x,y coordinates of bounding box for the obj
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

#draw the bounding box of the face along with associated probability
        text = "{:2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10

        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        cv2.destroyAllWindows()
        vs.stop()
