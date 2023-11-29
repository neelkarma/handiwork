"""
ACTIONS
-------

Pinch your index finger and thumb together to start a gesture, and drag your hand in a direction to perform an action.

Note: The actions performed only work if being run in the Sway WM, on Linux. Feel free to modify the code (specifically the ACTIONS constant) to suit your needs.
"""

import math
import subprocess

import cv2
import mediapipe as mp

from common.hands import dist_between, fraction_to_pixels, get_pinch_pointer, is_pinch

mp_hands = mp.solutions.hands

ACTIONS = {
    "r": ("prev desktop", lambda: subprocess.run(["swaymsg", "workspace", "prev"])),
    "l": ("next desktop", lambda: subprocess.run(["swaymsg", "workspace", "next"])),
    "d": ("hide all windows", lambda: subprocess.run(["swaymsg", "workspace", "11"])),
    "u": ("exit", lambda: exit(0)),
}


hands = mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
)


def get_direction(ox, oy, px, py):
    angle = math.atan2(py - oy, px - ox)
    dir_num = int((angle + (5 * math.pi / 4)) / (math.pi / 2))
    direction = ["l", "u", "r", "d", "l"][dir_num]
    return direction


# Initialize OpenCV
cap = cv2.VideoCapture(2)  # Use 0 for the default camera
gesture_origin = None
last_pointer = None
was_triggered = False
confirm_frame = 5
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
                if was_triggered:
                    continue
                last_pointer = get_pinch_pointer(landmarks)
                last_pointer = fraction_to_pixels(
                    frame, last_pointer[0], last_pointer[1]
                )
                if gesture_origin is None:
                    gesture_origin = last_pointer

                ox, oy = gesture_origin
                px, py = last_pointer

                angle = math.atan2(py - oy, px - ox)
                direction = get_direction(ox, oy, px, py)
                action = ACTIONS[direction][0]

                cv2.putText(
                    frame, action, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2
                )
                frame = cv2.line(
                    frame,
                    (int(gesture_origin[0]), int(gesture_origin[1])),
                    (
                        int(gesture_origin[0] + 200 * math.cos(angle)),
                        int(gesture_origin[1] + 200 * math.sin(angle)),
                    ),
                    (255, 0, 0),
                    10,
                )

                frame = cv2.line(
                    frame,
                    (int(gesture_origin[0]), int(gesture_origin[1])),
                    (int(last_pointer[0]), int(last_pointer[1])),
                    (0, 0, 255),
                    10,
                )

                confirm_frames = 5

                if dist_between(ox, oy, px, py) > 200:
                    direction = get_direction(ox, oy, px, py)

                    ACTIONS[direction][1]()
                    print(direction)

                    gesture_origin = None
                    last_pointer = None
                    was_triggered = True
                    continue

            elif gesture_origin is not None:
                confirm_frames -= 1
                if confirm_frames > 0:
                    continue

                print("cancel")

                gesture_origin = None
                last_pointer = None
                confirm_frames = 5
            elif was_triggered:
                was_triggered = False

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
