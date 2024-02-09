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

    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    for meth in methods:
        method = eval(meth)
        res = cv2.matchTemplate(gray_frame, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        h, w = template.shape
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
        cv2.putText(frame, meth, (top_left[0], top_left[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Live Stream", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.release()
cv2.destroyAllWindows()