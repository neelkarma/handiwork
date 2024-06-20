"""
GUN
-------

Gun.
"""

import math

import cv2
import mediapipe as mp

from common.hands import atan2, fraction_to_pixels

FRAMES_TO_RELOAD = 15
MAX_AMMO = 10
RANGE = 300

mp_hands = mp.solutions.hands

def is_gun(landmarks):
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_cmc = landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
    middle_dip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    middle_pip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_dip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
    ring_pip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_dip = landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]
    pinky_pip = landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]

    index_dir = atan2(index_mcp, index_tip)
    middle_dir = atan2(middle_pip, middle_dip)
    ring_dir = atan2(ring_pip, ring_dip)
    pinky_dir = atan2(pinky_pip, pinky_dip)
    thumb_dir = atan2(thumb_cmc, thumb_tip)

    right_angled_gun = abs(thumb_dir - index_dir) < (3 / 4) * math.pi
    three_fingers_behind = all(abs(finger_dir - index_dir) > 3 * math.pi / 4 for finger_dir in (middle_dir, ring_dir, pinky_dir))

    if not (right_angled_gun and three_fingers_behind):
        return None

    is_shooting = abs(thumb_dir - index_dir) < (3 / 16) * math.pi
    return index_dir, is_shooting

hands = mp_hands.Hands(
    max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7
)


# Initialize OpenCV
cap = cv2.VideoCapture(2)  # Use 0 for the default camera
ammo = MAX_AMMO
just_shot = False
reload_frames = FRAMES_TO_RELOAD

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    gun_present = False

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the image
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            gun = is_gun(landmarks)
            if gun is not None:
                gun_present = True
                index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                direction, is_shooting = gun
                gun_x, gun_y = fraction_to_pixels(frame, index_tip.x, index_tip.y)
                if ammo > 0:
                    cv2.line(
                        frame,
                        (int(gun_x), int(gun_y)),
                        (int(gun_x + math.cos(direction) * RANGE), int(gun_y + math.sin(direction) * RANGE)),
                        (0, 0, 255) if is_shooting else (255, 0, 0),
                        3
                    )

                if is_shooting and not just_shot and ammo > 0:
                    just_shot = True
                    ammo -= 1
                elif not is_shooting and just_shot:
                    just_shot = False

                cv2.putText(
                    frame,
                    str(ammo) if ammo > 0 else "reload!",
                    (int(gun_x + math.cos(direction) * 20 + math.cos(direction + math.pi / 2) * 20), int(gun_y + math.sin(direction) * 20 + math.sin(direction + math.pi / 2) * 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255) if ammo == 0 else (255, 255, 255),
                    3
                )

    if gun_present:
        reload_frames = FRAMES_TO_RELOAD
    elif ammo < MAX_AMMO:
        cv2.putText(
            frame,
            f"reloading... ({int((1 - reload_frames / FRAMES_TO_RELOAD) * 100)}%)",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )
        reload_frames = reload_frames - 1
        if reload_frames == 0:
                reload_frames = FRAMES_TO_RELOAD
                ammo = MAX_AMMO


    # Display the resulting frame
    cv2.imshow("handiwork", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the VideoCapture and close all windows
cap.release()
cv2.destroyAllWindows()
