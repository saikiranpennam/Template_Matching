import cv2

# Step 1: Initialize the camera and capture a live stream
cap = cv2.VideoCapture(0)

# Step 2: Capture a frame and select ROIs for each template
while True:
    ret, frame = cap.read()
    rois = []

    # Show the live stream and wait for key press
    cv2.imshow("Live Stream", frame)
    key = cv2.waitKey(1)

    # Press 'c' to capture ROI and 'q' to quit ROI selection
    if key == ord('c'):
        # Select ROI and append to list of ROIs
        roi = cv2.selectROI(frame)
        rois.append(roi)
        cv2.destroyAllWindows()
    elif key == ord('q'):
        cv2.destroyAllWindows()
        break

# Step 3: Switch to recognition mode and perform template matching on each ROI
templates = []
threshold = 0.7

for roi in rois:
    # Extract template from ROI and convert to grayscale
    template = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Append template to list of templates
    templates.append(template_gray)

while True:
    # Step 4: Continuously capture video frames and perform template matching
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform template matching on each template
    for i, template in enumerate(templates):
        res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # Draw rectangle around matched region if above threshold
        if max_val > threshold:
            x, y, w, h = rois[i]
            cv2.rectangle(frame, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Template {i}", max_loc, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Recognition Mode", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()