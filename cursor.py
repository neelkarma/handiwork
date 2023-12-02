"""
CURSOR
----

Control the mouse cursor with your hands! Pinch to press the left mouse button.

Note: Requires the `pynput` package to be installed
"""

import cv2
import mediapipe as mp
from pynput.mouse import Button, Controller

from common.hands import fraction_to_pixels, get_pinch_pointer, is_pinch

mp_hands = mp.solutions.hands

CURSOR_MIN_X, CURSOR_MIN_Y, CURSOR_MAX_X, CURSOR_MAX_Y = 2225, 41, 3819, 980


hands = mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
)
mouse = Controller()

# Initialize OpenCV
cap = cv2.VideoCapture(2)  # Use 0 for the default camera
is_pressing = False


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

            px, py = get_pinch_pointer(landmarks)

            mouse.position = (
                CURSOR_MIN_X + px * (CURSOR_MAX_X - CURSOR_MIN_X),
                CURSOR_MIN_Y + py * (CURSOR_MAX_Y - CURSOR_MIN_Y),
            )

            is_pinching = is_pinch(landmarks)

            if is_pinching and not is_pressing:
                is_pressing = True
                mouse.press(Button.left)
            elif not is_pinch(landmarks) and is_pressing:
                is_pressing = False
                mouse.release(Button.left)

            px, py = fraction_to_pixels(frame, px, py)
            cv2.circle(
                frame,
                (int(px), int(py)),
                5,
                (0, 255, 0) if is_pinching else (255, 0, 0),
                -1,
            )

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
