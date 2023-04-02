import cv2
import numpy as np

# Step 1: Select ROIs and save as templates
cap = cv2.VideoCapture(0)
templates = []
template_names = ["template1", "template2", "template3"]
template_idx = 0
num_objects = len(template_names)
while template_idx < num_objects:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Live Stream", frame)
    if cv2.waitKey(1) == ord('q'): # Press 'q' to exit
        break
    if cv2.waitKey(1) == ord('c'): # Press 'c' to capture ROI and save as template
        x, y, w, h = cv2.selectROI("Live Stream", frame, fromCenter=False)
        template = frame[y:y+h, x:x+w]
        templates.append(template)
        cv2.imshow(template_names[template_idx], template)
        cv2.imwrite(f"{template_names[template_idx]}.jpg", template)
        template_idx += 1

# Step 2: Perform template matching on live stream
if len(templates) == num_objects:
    threshold = 0.6
    k = 10  # examine every 10th video frame
    methods = [
        ('TM_CCOEFF', cv2.TM_CCOEFF)
        # ('TM_CCOEFF_NORMED', cv2.TM_CCOEFF_NORMED),
        # ('TM_CCORR', cv2.TM_CCORR),
        # ('TM_CCORR_NORMED', cv2.TM_CCORR_NORMED),
        # ('TM_SQDIFF', cv2.TM_SQDIFF),
        # ('TM_SQDIFF_NORMED', cv2.TM_SQDIFF_NORMED)
    ]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if cv2.waitKey(1) == ord('q'): # Press 'q' to exit
            break
        if k == 0:
            for i, template in enumerate(templates):
                for method_name, method in methods:
                    res = cv2.matchTemplate(frame, template, method)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                        top_left = min_loc
                    else:
                        top_left = max_loc
                    # template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                    w, h, _ = template.shape[::-1]
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    if max_val > threshold:
                        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                        cv2.putText(frame, f"{template_names[i]}", top_left,
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Live Stream", frame)
            k = 10  # reset counter
        else:
            k -= 1

    cap.release()
    cv2.destroyAllWindows()
else:
    print(f"Only {len(templates)} objects were captured, need {num_objects} objects.")