import cv2
import numpy as np

cap = cv2.VideoCapture(0)
templates = []
n = 3
i = 0
method = cv2.TM_CCOEFF_NORMED
threshold = 0.7
capture_mode = True
while True:
    ret, frame= cap.read()
    if not ret:
        print("Error capturing frame")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame)
    if capture_mode:
        if cv2.waitKey(1) == ord(' '):
            r = cv2.selectROI(frame)
            template = gray[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
            templates.append(template)
            i += 1
            if i == n:
                capture_mode = False
    else:
        for j, template in enumerate(templates):
            res= cv2.matchTemplate(gray, template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val >= threshold:
                w,h = template.shape[::-1]
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(frame, f"Object {j}", top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()