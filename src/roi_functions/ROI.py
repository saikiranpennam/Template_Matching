import cv2

# Initialize the camera
cap = cv2.VideoCapture(0)

# Define the function to capture a frame and select a region of interest
def capture_template():
    _, frame = cap.read()
    # print(frame.shape)
    cv2.imshow('Select ROI', frame)
    roi = cv2.selectROI(frame)
    template = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    cv2.destroyAllWindows()
    return template

# Capture templates for three objects
templates = []
for i in range(3):
    print(f'Capture template for object {i+1}')
    template = capture_template()
    templates.append(template)

# Switch to recognition mode
while True:
    _, frame = cap.read()

    # Perform template matching for each object
    for i, template in enumerate(templates):
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        w, h = template.shape[::-1]

        # If match score exceeds threshold, identify the object in the video by name
        if max_val > 0.8: # threshold is 0.8
            cv2.rectangle(frame, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 0), 2)
            cv2.putText(frame, f'Object {i+1}', max_loc, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Recognition Mode', frame)

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()