"""
SLIDER
------

Pinch and drag the slider handle to change the volume.

Note: This script requires that PulseAudio and `pamixer` is installed. You can change what the slider does by modifying the `callback` argument in the Slider constructor.
"""

import cv2
import mediapipe as mp

from common.hands import (
    EasyHandLandmarker,
    dist_between,
    draw_landmarks,
    fraction_to_pixels,
    get_pinch_pointer,
    is_pinch,
    pixels_to_fraction,
)
from common.volume import get_volume, set_volume


class Slider:
    def __init__(self, x, y1, y2, callback, value=0):
        self.x = x
        self.y1 = y1
        self.y2 = y2
        self.callback = callback
        self.value = value
        self.is_active = False

    def get_handle_coords(self):
        return self.x, self.y1 + int((self.y2 - self.y1) * self.value)

    def update(self, frame, landmarks):
        if is_pinch(landmarks):
            px, py = get_pinch_pointer(landmarks)
            hx, hy = pixels_to_fraction(frame, *self.get_handle_coords())

            if dist_between(px, py, hx, hy) <= 0.06:
                self.is_active = True

            if self.is_active:
                px, py = fraction_to_pixels(frame, px, py)
                self.value = (py - self.y1) / (self.y2 - self.y1)
                self.value = max(0, min(1, self.value))
                self.callback(self.value)
        elif self.is_active:
            self.is_active = False

    def render(self, frame):
        cv2.putText(
            frame,
            f"Volume: {int(self.value * 100)}%",
            (10, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (255, 0, 0),
            2,
        )
        cv2.line(
            frame,
            (self.x, self.y1),
            (self.x, self.y2),
            (255, 255, 255),
            5,
        )

        cv2.circle(
            frame,
            self.get_handle_coords(),
            10,
            (0, 0, 255) if self.is_active else (255, 0, 0),
            -1,
        )


landmarker = EasyHandLandmarker()

# Initialize OpenCV
cap = cv2.VideoCapture(2)  # Use 0 for the default camera
slider = Slider(
    100, 400, 100, lambda val: set_volume(int(val * 100)), get_volume() / 100
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Process the frame with MediaPipe Hands
    landmarker.process_frame(frame)
    results = landmarker.get_latest_result()

    if results:
        draw_landmarks(frame, results)
        for landmarks in results.hand_landmarks:
            slider.update(frame, landmarks)

    slider.render(frame)
    # Display the resulting frame
    cv2.imshow("handiwork", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the VideoCapture and close all windows
cap.release()
cv2.destroyAllWindows()
landmarker.close()
