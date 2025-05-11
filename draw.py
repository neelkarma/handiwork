"""
DRAW
----

Pinch your index finger and thumb together to start drawing, and move your hand to draw on the screen.

Press 'r' to reset.
"""

import cv2
import mediapipe as mp

from common.hands import (
    EasyHandLandmarker,
    draw_landmarks,
    fraction_to_pixels,
    get_pinch_pointer,
    is_pinch,
)

mp_hands = mp.solutions.hands


landmarker = EasyHandLandmarker(num_hands=1)

# Initialize OpenCV
cap = cv2.VideoCapture(2)  # Use 0 for the default camera
was_drawing = False
drawing = [[]]

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
                was_drawing = True
                pointer = get_pinch_pointer(landmarks)
                pointer = fraction_to_pixels(frame, pointer[0], pointer[1])
                drawing[-1].append((int(pointer[0]), int(pointer[1])))
            elif was_drawing:
                was_drawing = False
                drawing.append([])

    # draw the strokes
    for stroke in drawing:
        for i in range(1, len(stroke)):
            x1, y1 = stroke[i - 1]
            x2, y2 = stroke[i]
            image = cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)

    # Display the resulting frame
    cv2.imshow("handiwork", frame)

    # handle keyboard input
    key = cv2.waitKey(1)
    if key & 0xFF == ord("r"):
        drawing = [[]]
    if key & 0xFF == ord("q"):
        break

# Release the VideoCapture and close all windows
cap.release()
cv2.destroyAllWindows()
landmarker.close()
