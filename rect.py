"""
RECTANGLES
----------

Pinch your index finger and thumb with both hands at the same point to start drawing a rectangle, and drag them outwards to expand it.

Press 'r' to reset.
"""

import cv2
import mediapipe as mp

from common.hands import dist_between, fraction_to_pixels, get_pinch_pointer, is_pinch

mp_hands = mp.solutions.hands


hands = mp_hands.Hands(
    max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7
)

# Initialize OpenCV
cap = cv2.VideoCapture(2)
is_active = False
last_rect = None
rects = []

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
        pointers = []
        for landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the image
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            if is_pinch(landmarks):
                pointers.append(get_pinch_pointer(landmarks))

        if len(pointers) == 2:
            p1, p2 = pointers
            p1x, p1y = p1
            p2x, p2y = p2

            if not is_active:
                # check if the four fingers are close enough
                if dist_between(p1x, p1y, p2x, p2y) <= 0.06:
                    is_active = True

            if is_active:
                p1x, p1y = map(int, fraction_to_pixels(frame, p1x, p1y))
                p2x, p2y = map(int, fraction_to_pixels(frame, p2x, p2y))
                cv2.rectangle(frame, (p1x, p1y), (p2x, p2y), (0, 0, 255), 5)
                last_rect = p1x, p1y, p2x, p2y

        elif is_active:
            is_active = False
            rects.append(last_rect)

    for x1, y1, x2, y2 in rects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)

    # Display the resulting frame
    cv2.imshow("handiwork", frame)

    key = cv2.waitKey(1)
    # reset if 'r' is pressed
    if key & 0xFF == ord("r"):
        rects = []
    # Break the loop if 'q' is pressed
    if key & 0xFF == ord("q"):
        break

# Release the VideoCapture and close all windows
cap.release()
cv2.destroyAllWindows()
