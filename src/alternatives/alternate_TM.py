import cv2

cap = cv2.VideoCapture(0)

# Step 1: Select ROI and save as template
while True:
    ret, frame = cap.read()
    cv2.imshow("Live Stream", frame)
    if cv2.waitKey(1) == ord('q'): # Press 'q' to exit
        break
    if cv2.waitKey(1) == ord('c'): # Press 'c' to capture ROI and save as template
        x, y, w, h = cv2.selectROI(frame)
        template = frame[y:y+h, x:x+w]
        cv2.imshow("ROI", template)
        cv2.imwrite("template.jpg", template)
        break

# Step 2: Perform template matching on live stream
template = cv2.imread("template.jpg", cv2.IMREAD_GRAYSCALE)
threshold = 0.8

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(gray_frame, template, cv2.TM_SQDIFF_NORMED)

    '''
    The best template for faces is TM_CCOEFF and the worst is TM-SQDIFF ,TM_CORR
    '''
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if max_val > threshold:
        x, y = max_loc
        w, h = template.shape[::-1]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Object Detected", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Live Stream", frame)

    if cv2.waitKey(1) == ord('q'): # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()