import cv2

cap = cv2.VideoCapture(0)

# Step 1: Select ROIs and save as templates
templates = []
while len(templates) < 3:
    ret, frame = cap.read()
    cv2.imshow("Live Stream", frame)
    if cv2.waitKey(1) == ord('q'): # Press 'q' to exit
        break
    if cv2.waitKey(1) == ord('c'): # Press 'c' to capture ROI and save as template
        x, y, w, h = cv2.selectROI(frame)
        template = frame[y:y+h, x:x+w]
        templates.append(template)
        cv2.imshow("ROI", template)
        cv2.imwrite(f"template{len(templates)}.jpg", template)

# Step 2: Perform template matching on live stream
threshold = 0.8

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for i, template in enumerate(templates):
        res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val > threshold:
            x, y = max_loc
            w, h = template.shape[::-1]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Object {i+1} Detected", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Live Stream", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cap.destroyAllWindows()