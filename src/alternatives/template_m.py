import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import glob as gb

objects = {}
NO_OF_OBJECTS=0
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('outputmultiobjs.avi', fourcc, 20.0, (640,480))

def readAnnotation(path):
    files = gb.glob(path + "/*.jpg")
    for f in files:
        obj_name = f.split('_')[0].split('\\')[1]
        img = cv.imread(f,0)
        if obj_name in objects:
            objects[obj_name].append(img)
        else:
            objects[obj_name] = [img]

    NO_OF_OBJECTS = len(objects)

# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED','cv.TM_CCOEFF']
# font
font = cv.FONT_HERSHEY_SIMPLEX
# fontScale
fontScale = 1
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2


readAnnotation("Templates")
cap = cv.VideoCapture(0)
if cap.isOpened():
    ret , frame = cap.read()
    if ret:
        rgb = frame
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    i = 0
    skip_frames = 2
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
          if i%skip_frames == 0 or i == 0:
            rgb = frame
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            img = frame.copy()
            for ts in objects:
              for t in objects[ts]:
                w, h = t.shape[::-1]

                method = eval(methods[5]) # 0=cv.TM_CCOEFF_NORMED, 1=cv.TM_CCORR, 2=cv.TM_CCORR_NORMED, 3=cv.TM_SQDIFF, 4=cv.TM_SQDIFF_NORMED, 5=cv.TM_CCOEFF
                #ret, th1 = cv.threshold(cv.GaussianBlur(t,(3,3),0), 127, 255, cv.THRESH_BINARY)
                # Apply template Matching
                #t = cv.GaussianBlur(t,(3,3),0)
                #img = cv.GaussianBlur(img, (3, 3), 0)
                res = cv.matchTemplate(img, t, method)
                min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
                print(max_val)
                detected = False
                threshold = 0.75
                # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                    top_left = min_loc
                    #top_score = min_val
                    #threshold =
                else:
                    top_left = max_loc
                    #top_score = max_val
                bottom_right = (top_left[0] + w, top_left[1] + h)

                if max_val > threshold:
                    detected = True
                loc = np.where(res >= threshold)
                #for pt in zip(*loc[::-1]):
                #    cv.rectangle(rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
                #    img = cv.putText(img, ts, top_left, font,
                #                     fontScale, color, thickness, cv.LINE_AA)
                #out.write(rgb)


                # for l in loc:
                #     if len(l) != 0:
                #         detected = True
                #     else:
                #         detected = False

                if detected:

                    cv.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)
                    # Using cv2.putText() method
                    img = cv.putText(img, ts, top_left, font,
                                     fontScale, color, thickness, cv.LINE_AA)
                cv.imshow("Output", img)
          i = i + 1


          # Exit if ESC pressed
          k = cv.waitKey(1) & 0xff
          if k == 27:
              out.release()
              break