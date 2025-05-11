"""
MUSIC
-----

Tap your index finger and thumb together:
- Once to play/pause
- Twice to skip to the next song
- Three times to skip to the previous song

Hold your index finger and thumb together and drag up or down to adjust volume.

Note: This script requires that:
- You are on Linux
- You have `pamixer` and `playerctl` installed
"""

import subprocess
import time

import cv2
import mediapipe as mp

from common.hands import (
    EasyHandLandmarker,
    dist_between,
    draw_landmarks,
    fraction_to_pixels,
    get_pinch_pointer,
    is_pinch,
)
from common.volume import get_volume, set_volume

mp_hands = mp.solutions.hands

TAP_DELAY_SECONDS = 0.4
VOLUME_SENSITIVITY = 0.3
START_VOLUME_ADJUST_DIST = 50

landmarker = EasyHandLandmarker()

# Initialize OpenCV
cap = cv2.VideoCapture(2)  # Use 0 for the default camera
gesture_origin = None
is_dragging = False
original_volume = None
num_taps = 0
last_tap = None

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
                pointer = get_pinch_pointer(landmarks)
                pointer = fraction_to_pixels(frame, pointer[0], pointer[1])

                if gesture_origin is None:
                    gesture_origin = pointer
                    num_taps += 1
                    last_tap = time.time()
                elif (
                    not is_dragging
                    and dist_between(*pointer, *gesture_origin)
                    > START_VOLUME_ADJUST_DIST
                    or time.time() > TAP_DELAY_SECONDS + (last_tap or time.time())
                ):
                    is_dragging = True
                    last_tap = None
                    num_taps = 0
                    original_volume = get_volume()

                if is_dragging:
                    dy = gesture_origin[1] - pointer[1]
                    set_volume(original_volume + dy * VOLUME_SENSITIVITY)
            else:
                gesture_origin = None
                is_dragging = False

    if last_tap is not None and gesture_origin is None:
        if time.time() > TAP_DELAY_SECONDS + last_tap or num_taps >= 3:
            if num_taps == 1:
                print("play/pause")
                subprocess.run(["playerctl", "play-pause"])
            elif num_taps == 2:
                print("next")
                subprocess.run(["playerctl", "next"])
            elif num_taps == 3:
                print("prev")
                subprocess.run(["playerctl", "previous"])

            num_taps = 0
            last_tap = None

    if num_taps == 1:
        cv2.putText(
            frame, "play/pause", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2
        )
    elif num_taps == 2:
        cv2.putText(frame, "next", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
    elif num_taps == 3:
        cv2.putText(frame, "prev", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
    elif is_dragging:
        cv2.putText(
            frame,
            "adjusting volume",
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
