# Step 1: Initialize the camera and capture live stream
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow("Live Stream", frame)
    if cv2.waitKey(1) == 27: # Press 'esc' to exit
        break

# Step 2: Press key to capture a frame of video
    if cv2.waitKey(1) == ord('c'): # Press 'c' to capture frame
        break

# Step 3: Drawing a rectangle around the object of interest in the captured frame
    x, y, w, h = cv2.selectROI(frame)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow("ROI", frame[y:y+h, x:x+w])

    # Step 4: Saving ROI as template image
    cv2.imwrite("template.jpg", frame[y:y+h, x:x+w])

# Step 5: Switching to Recognition Mode
    template1 = cv2.imread("/Users/saikiranpennam/PycharmProjects/Assingment2_CS510/template.jpg", cv2.IMREAD_GRAYSCALE)
# template2 = cv2.imread("template2.jpg", cv2.IMREAD_GRAYSCALE)
# template3 = cv2.imread("template3.jpg", cv2.IMREAD_GRAYSCALE)

# Step 6: Continuously capturing video frames and performing template matching
    threshold = 0.8
# Converting the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform Template Matching
    res1 = cv2.matchTemplate(gray_frame, template1, cv2.TM_CCOEFF_NORMED)
    # res2 = cv2.matchTemplate(gray_frame, template2, cv2.TM_CCOEFF_NORMED)
    # res3 = cv2.matchTemplate(gray_frame, template3, cv2.TM_CCOEFF_NORMED)

    min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
    # min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)
    # min_val3, max_val3, min_loc3, max_loc3 = cv2.minMaxLoc(res3)

# Step 7: Finding the location of the best match and comparing the match score to a threshold
    if max_val1 > threshold:
        cv2.rectangle(frame, max_loc1, (max_loc1[0] + w, max_loc1[1] + h), (0, 255, 0), 2)
        cv2.putText(frame, "Object 1", max_loc1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
'''
    if max_val2 > threshold:
        cv2.rectangle(frame, max_loc2, (max_loc2[0] + w, max_loc2[1] + h), (255, 0, 0), 2)
        cv2.putText(frame, "Object 2", max_loc2, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if max_val3 > threshold:
        cv2.rectangle(frame, max_loc2, (max_loc2[0] + w, max_loc2[1] + h), (0, 0, 255), 2)
        cv2.putText(frame, "Object 3", max_loc2, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,255), 2)
'''
cap.release()
cv2.destroyAllWindows()