import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    #Capture Frame-by-frame
    #print('test')
    ret, frame = cap.read()

    grayscaleclr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
    e_kernel = np.ones((2, 2), np.uint8)
    d_kernel = np.ones((4, 4), np.uint8)
    blurred = cv2.blur(grayscaleclr, (4, 4))
    ret, nt = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    #thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                               #cv2.THRESH_BINARY, 11, 2)
    #erode = cv2.erode(thresh, e_kernel, iterations=1)
    #dilate = cv2.dilate(erode, d_kernel, iterations=1)

    im2, contours, hierarchy = cv2.findContours(nt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    findContours = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    #Show frames
    #cv2.imshow('original', frame)
    #cv2.imshow('gray-scale', blurred)
    cv2.imshow('frame1', nt)
    cv2.imshow('frame2', findContours)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#when everything done, release the capture
cap.release()
cv2.destroyAllWindows()