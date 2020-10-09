import argparse
import time
import cv2

capture = cv2.VideoCapture(0)

# frame 1
success, frame1 = capture.read()
# frame 2
success, frame2 = capture.read()

while capture.isOpened():
    # diffrence beetween frames
    diff = cv2.absdiff(frame1, frame2)
    # convert to gray to find contures
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # blurring(source,(kernel size),sigma x value)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # finding the thershold(source,thershold value,maximum thershold value,type)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    # filling the holes(source,kernel size,iterations)
    dilated = cv2.dilate(thresh, None, iterations=3)
    # find the contour,result(source,mode,method)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # apllying method to get contours(source,source,contour id,color,thickness)
    # cv2.drawContours(frame1,contour,-1,(0,255,0),2)

    # draw the rectangles
    for contour in contours:
        # save the contours boundingRect() returns x,y,w,h
        (x, y, w, h) = cv2.boundingRect(contour)
        # if contour area small dont draw
        if cv2.contourArea(contour) < 700:
            continue

        # draw the rectangle(source,(point1),(point2),(color),(thickness)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # put text(source,(msg),(where),(font),(font size),(color),(thicness)
        cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

    cv2.imshow('Video', frame1)
    # reassign value to compare the frame 2
    frame1 = frame2
    # read the frame
    ret, frame2 = capture.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
capture.release()
