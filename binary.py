"""
BINARY COUNTING
-------

Count in binary with your hands! Each finger represents a bit - the pinky represents 2 ** 0, the ring finger represents 2 ** 1, and so on.
"""

import math

import cv2
import mediapipe as mp

from common.hands import EasyHandLandmarker, draw_landmarks, fingers_are_up

mp_hands = mp.solutions.hands


landmarker = EasyHandLandmarker()

# Initialize OpenCV
cap = cv2.VideoCapture(2)  # Use 0 for the default camera

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
        total = 0
        for hand, landmarks in enumerate(results.hand_landmarks):
            for i, is_up in enumerate(fingers_are_up(landmarks)):
                total <<= 1
                if is_up:
                    total += 1

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
landmarker.close()
