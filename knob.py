"""
KNOB
-------

Pinch fingers together and turn your hand.
"""

import math

import cv2
import mediapipe as mp

from common.hands import (
    EasyHandLandmarker,
    draw_landmarks,
    is_pinch,
)

HandLandmark = mp.solutions.hands.HandLandmark

LINE_LENGTH = 100


landmarker = EasyHandLandmarker()


def calc_angle(landmarks):
    index = landmarks[HandLandmark.INDEX_FINGER_TIP]
    pinky = landmarks[HandLandmark.PINKY_TIP]
    return math.atan2(pinky.y - index.y, pinky.x - index.x)


# Initialize OpenCV
cap = cv2.VideoCapture(2)  # Use 0 for the default camera
prev_angle = 0
angle = 0
start_angle = None
is_exit = False

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
            if is_pinch(landmarks):
                if start_angle is None:
                    prev_angle = angle
                    start_angle = calc_angle(landmarks)
                else:
                    angle_delta = start_angle - calc_angle(landmarks)
                    angle = prev_angle + angle_delta
            elif start_angle is not None:
                prev_angle = None
                start_angle = None

    h, w, _ = frame.shape
    x, y = w // 2, h // 2
    cv2.line(
        frame,
        (x, y),
        (
            int(x + LINE_LENGTH * math.sin(angle)),
            int(y + LINE_LENGTH * math.cos(angle)),
        ),
        (255, 0, 0) if start_angle is None else (0, 0, 255),
        thickness=10,
    )

    if is_exit:
        break

    # Display the resulting frame
    cv2.imshow("handiwork", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the VideoCapture and close all windows
cap.release()
cv2.destroyAllWindows()
