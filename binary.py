"""
BINARY COUNTING
-------

Count in binary with your hands! Each finger represents a bit - the pinky represents 2 ** 0, the ring finger represents 2 ** 1, and so on.
"""

import math

import cv2
import mediapipe as mp

from common.hands import fingers_are_up

mp_hands = mp.solutions.hands


hands = mp_hands.Hands(
    max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7
)


# Initialize OpenCV
cap = cv2.VideoCapture(2)  # Use 0 for the default camera

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
        total = 0
        for hand, landmarks in enumerate(results.multi_hand_landmarks):
            # Draw hand landmarks on the image
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            for i, is_up in enumerate(fingers_are_up(landmarks)):
                if is_up:
                    total += 2 ** (i + hand * 5)

        cv2.putText(
            frame,
            f"{bin(total)} = {total}",
            (10, 50),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (255, 0, 0),
            2,
        )

    # Display the resulting frame
    cv2.imshow("handiwork", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the VideoCapture and close all windows
cap.release()
cv2.destroyAllWindows()
