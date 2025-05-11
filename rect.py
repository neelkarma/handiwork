"""
RECTANGLES
----------

Pinch your index finger and thumb with both hands at the same point to start drawing a rectangle, and drag them outwards to expand it.

Press 'r' to reset.
"""

import cv2

from common.hands import (
    EasyHandLandmarker,
    dist_between,
    draw_landmarks,
    fraction_to_pixels,
    get_pinch_pointer,
    is_pinch,
)

landmarker = EasyHandLandmarker(num_hands=2)

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

    # Process the frame with MediaPipe Hands
    landmarker.process_frame(frame)
    results = landmarker.get_latest_result()

    if results:
        pointers = []
        draw_landmarks(frame, results)
        for landmarks in results.hand_landmarks:
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
landmarker.close()
