import math
import subprocess

import cv2
import mediapipe as mp

from common import (
    dist_between,
    fraction_to_pixels,
    get_pinch_pointer,
    is_pinch,
    pixels_to_fraction,
)

mp_hands = mp.solutions.hands


hands = mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
)


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


def set_volume(value):
    subprocess.run(["pamixer", "--set-volume", str(int(value * 100))])


def get_volume():
    return int(
        subprocess.run(
            ["pamixer", "--get-volume"], capture_output=True, text=True
        ).stdout.strip("\n")
    )


# Initialize OpenCV
cap = cv2.VideoCapture(2)  # Use 0 for the default camera
slider = Slider(100, 400, 100, set_volume, get_volume() / 100)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the image
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
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
