"""
KNOB
-------

Pinch fingers together and turn your hand.
"""

import math

import cv2
import mediapipe as mp

from common.hands import is_pinch

LINE_LENGTH = 100

mp_hands = mp.solutions.hands


hands = mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
)

def calc_angle(landmarks):
    index = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    pinky = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
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

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the image
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

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

    height, width, _ = frame.shape
    cv2.line(
        frame,
        (int(width / 2), int(height / 2)),
        (int(width / 2 + LINE_LENGTH * math.sin(angle)), int(height / 2 + LINE_LENGTH * math.cos(angle))),
        (255, 0, 0) if start_angle is None else (0, 0, 255),
        thickness=10
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
