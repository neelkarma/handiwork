"""
MUSIC
-----

Tap your index finger and thumb together:
- Once to play/pause
- Twice to skip to the next song
- Three times to skip to the previous song

Note: This script requires that:
- You are listening to music on Spotify Desktop
- You are on Linux
- You have the `dbus-python` package installed
- You have `pamixer` installed
"""

import time

import cv2
import dbus
import mediapipe as mp

from common.hands import dist_between, fraction_to_pixels, get_pinch_pointer, is_pinch
from common.volume import get_volume, set_volume

mp_hands = mp.solutions.hands

TAP_DELAY_SECONDS = 0.4
VOLUME_SENSITIVITY = 0.3
START_VOLUME_ADJUST_DIST = 50


hands = mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
)


# Initialize OpenCV
cap = cv2.VideoCapture(2)  # Use 0 for the default camera
gesture_origin = None
is_dragging = False
original_volume = None
num_taps = 0
last_tap = None

# initialise dbus
session = dbus.SessionBus()
spotify = session.get_object(
    "org.mpris.MediaPlayer2.spotify", "/org/mpris/MediaPlayer2"
)
player = dbus.Interface(spotify, "org.mpris.MediaPlayer2.Player")

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
                    set_volume((original_volume + dy * VOLUME_SENSITIVITY) / 100)
            else:
                gesture_origin = None
                is_dragging = False

    if last_tap is not None and gesture_origin is None:
        if time.time() > TAP_DELAY_SECONDS + last_tap or num_taps >= 3:
            if num_taps == 1:
                print("play/pause")
                player.PlayPause()
            elif num_taps == 2:
                print("next")
                player.Next()
            elif num_taps == 3:
                print("prev")
                player.Previous()

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
