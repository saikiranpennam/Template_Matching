import cv2
import numpy as np
import time
from threading import Thread

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Step 1: Select ROIs and save as templates
templates = []
template_names = ["template1", "template2", "template3"]
template_idx = 0

while template_idx < len(template_names):
    ret, frame = cap.read()
    cv2.imshow("Live Stream", frame)
    if cv2.waitKey(1) == ord('q'): # Press 'q' to exit
        break
    if cv2.waitKey(1) == ord('c'): # Press 'c' to capture ROI and save as template
        x, y, w, h = cv2.selectROI(frame)
        template = frame[y:y+h, x:x+w]
        # template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template = cv2.convertScaleAbs(template)
        templates.append(template)
        cv2.imshow(template_names[template_idx], template)
        cv2.imwrite(f"{template_names[template_idx]}.jpg", template)
        template_idx += 1

# Step 2: Define template matching function to be run on a separate thread
def match_template(template, frame, threshold, method_name, method_idx, results):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray_frame, template, method_idx)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val > threshold:
        results[method_name] = (True, max_val, max_loc)
    else:
        results[method_name] = (False, 0, (0, 0))

# Step 3: Define a function to run template matching on a single frame
def match_frame(frame, templates, threshold, methods):
    results = {}

    for template_idx, template in enumerate(templates):
        threads = []

        for method_name, method_idx in methods:
            thread = Thread(target=match_template, args=(template, frame, threshold, method_name, method_idx, results))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        for method_name, (is_detected, max_val, max_loc) in results.items():
            if is_detected:
                w, h = template.shape[::-1]
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(frame, f"Template {template_idx+1} using {method_name}: Object Detected", (top_left[0], top_left[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

# Step 4: Run template matching on every kth frame
k = 5
threshold = 0.8

methods = [
    ('TM_CCOEFF', cv2.TM_CCOEFF),
    ('TM_CCOEFF_NORMED', cv2.TM_CCOEFF_NORMED),
    ('TM_CCORR', cv2.TM_CCORR),
    ('TM_CCORR_NORMED', cv2.TM_CCORR_NORMED),
    ('TM_SQDIFF', cv2.TM_SQDIFF),
    ('TM_SQDIFF_NORMED', cv2.TM_SQDIFF_NORMED)
]

while True:
    for i in range(k):
        ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for i, template in enumerate(templates):
        for method_name, method in methods:
            res = cv2.matchTemplate(gray_frame, template, method)

            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                top_left = min_loc
            else:
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                top_left = max_loc

            w, h = template.shape[::-1]
            bottom_right = (top_left[0] + w, top_left[1] + h)

            if max_val > threshold:
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(frame, f"Template {i+1} using {method_name}: Object Detected", (top_left[0], top_left[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Live Stream", frame)

    if cv2.waitKey(10) == ord('q'): # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()