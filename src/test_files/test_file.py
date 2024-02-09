import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Object Recognition Demo')
        self.setGeometry(50, 50, 640, 480)
        self.video_feed = cv2.VideoCapture(0)
        self.template_images = []
        self.object_names = []
        self.match_threshold = 0.8
        self.init_ui()

    def init_ui(self):
        # Create buttons and labels for UI
        self.capture_button = QLabel('Press Spacebar to Capture')
        self.capture_button.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.capture_button)
        self.show()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            ret, frame = self.video_feed.read()
            if ret:
                # Display captured frame in UI
                self.display_frame(frame)
                # Let user select ROI and save as template
                roi = cv2.selectROI(frame)
                template_image = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
                self.template_images.append(template_image)
                self.object_names.append(QFileDialog.getSaveFileName(self, 'Save Template Image', filter='Images (*.png *.jpg)')[0])
                # Update UI to indicate template capture is complete
                self.capture_button.setText('Press Spacebar to Capture ({} templates captured)'.format(len(self.template_images)))
                if len(self.template_images) == 3:
                    # All templates have been captured, switch to recognition mode
                    self.recognition_mode()

    def recognition_mode(self):
        # Remove capture button from UI
        self.setCentralWidget(QLabel())
        self.show()
        # Initialize template matching parameters
        match_methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
        match_method = cv2.TM_CCOEFF_NORMED
        match_template_sizes = [(20, 20), (30, 30), (40, 40)]
        match_template_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        # Loop over frames and perform template matching
        while True:
            ret, frame = self.video_feed.read()
            if ret:
                # Loop over templates and perform template matching
                for i in range(len(self.template_images)):
                    template_image = self.template_images[i]
                    object_name = self.object_names[i]
                    for match_template_size in match_template_sizes:
                        # Resize template image to different sizes
                        template_image_resized = cv2.resize(template_image, match_template_size)
                        for match_template_threshold in match_template_thresholds:
                            # Try different thresholds for template matching
                            match_threshold = match_template_threshold
                            for match_method in match_methods:
                                # Try different matching methods
                                res = cv2.matchTemplate(frame, template_image_resized, match_method)
                                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                                if max_val >= match_threshold:
                                    # Object found, draw rectangle and label on frame
                                    top_left = max_loc
                                    bottom_right = (top_left[0] + match_template_size[0], top_left[1] + match_template_size[1])
                                    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                                    cv2.putText(frame, object_name, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                    # Display processed frame in UI
                                    self.display_frame(frame)

    def display_frame(self, frame):
        # Convert OpenCV BGR image to QImage for display in PyQt5 UI
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.setCentralWidget(QLabel(alignment=Qt.AlignCenter))
        self.centralWidget().setPixmap(pixmap)
        self.show()

    def closeEvent(self, event):
        self.video_feed.release()
