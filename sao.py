"""
SAO MENU
----

If you are a weeb you'll know how to use this.
"""

import math

import cv2
import mediapipe as mp

from common.hands import (
    EasyHandLandmarker,
    dist_between,
    draw_landmarks,
    fingers_are_up,
    fraction_to_pixels,
)

HandLandmark = mp.solutions.hands.HandLandmark

LEFT_HANDED = False  # Change to True if using left hand.


def is_initial_gesture(landmarks):
    middle_mcp = landmarks[HandLandmark.MIDDLE_FINGER_MCP]
    wrist = landmarks[HandLandmark.WRIST]
    angle = math.atan2(wrist.y - middle_mcp.y, wrist.x - middle_mcp.x)

    required_fingers_are_up = fingers_are_up(landmarks)[1:] == [
        True,
        True,
        False,
        False,
    ]

    has_correct_angle = (
        math.pi / 2 - math.pi / 6 < angle < 3 * math.pi / 2 + math.pi / 6
    )

    return required_fingers_are_up and has_correct_angle


def is_final_gesture(landmarks):
    index_tip = landmarks[HandLandmark.INDEX_FINGER_TIP]
    index_mcp = landmarks[HandLandmark.INDEX_FINGER_MCP]
    middle_tip = landmarks[HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = landmarks[HandLandmark.MIDDLE_FINGER_MCP]
    return index_tip.y > index_mcp.y and middle_tip.y > middle_mcp.y


def hand_fanned_right(landmarks):
    index = landmarks[HandLandmark.INDEX_FINGER_TIP]
    middle = landmarks[HandLandmark.MIDDLE_FINGER_TIP]
    ring = landmarks[HandLandmark.RING_FINGER_TIP]
    pinky = landmarks[HandLandmark.PINKY_TIP]
    wrist = landmarks[HandLandmark.WRIST]

    return all(point.x > wrist.x for point in [index, middle, ring, pinky])


def hand_fanned_left(landmarks):
    index = landmarks[HandLandmark.INDEX_FINGER_TIP]
    middle = landmarks[HandLandmark.MIDDLE_FINGER_TIP]
    ring = landmarks[HandLandmark.RING_FINGER_TIP]
    pinky = landmarks[HandLandmark.PINKY_TIP]
    wrist = landmarks[HandLandmark.WRIST]

    return all(point.x < wrist.x for point in [index, middle, ring, pinky])


def get_gesture_pointer(landmarks):
    index = landmarks[HandLandmark.INDEX_FINGER_TIP]
    # middle = landmarks[HandLandmark.MIDDLE_FINGER_TIP]
    # return (index.x + middle.x) / 2, (index.y + middle.y) / 2
    return index.x, index.y


landmarker = EasyHandLandmarker()

# Initialize OpenCV
cap = cv2.VideoCapture(2)  # Use 0 for the default camera

gesture_frames = 0
menu_delay_frames = None
menu_origin = None
menu_animation_frame = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Process the frame with MediaPipe Hands
    landmarker.process_frame(frame)
    results = landmarker.get_latest_result()

    menu_cursor = None

    if results:
        draw_landmarks(frame, results)
        for landmarks in results.hand_landmarks:
            if menu_origin is None:
                if is_initial_gesture(landmarks):
                    gesture_frames = 3
                elif gesture_frames > 0:
                    gesture_frames -= 1
                else:
                    gesture_origin = None

                if is_final_gesture(landmarks) and gesture_frames > 0:
                    gesture_frames = 0
                    menu_delay_frames = 4

                if menu_delay_frames is not None:
                    menu_delay_frames -= 1
                    if menu_delay_frames == 0:
                        menu_origin = get_gesture_pointer(landmarks)
                        menu_origin = fraction_to_pixels(frame, *menu_origin)
                        menu_delay_frames = None
            else:
                index_tip = landmarks[HandLandmark.INDEX_FINGER_TIP]
                menu_cursor = fraction_to_pixels(frame, index_tip.x, index_tip.y)

                if (hand_fanned_left if LEFT_HANDED else hand_fanned_right)(landmarks):
                    gesture_frames = 3
                elif gesture_frames > 0:
                    gesture_frames -= 1

                if (hand_fanned_right if LEFT_HANDED else hand_fanned_left)(
                    landmarks
                ) and gesture_frames > 0:
                    menu_origin = None
                    menu_animation_frame = 0

    if menu_origin is not None:
        if menu_animation_frame < 8:
            menu_animation_frame += 1
        for i in range(5):
            x, y = (
                menu_origin[0],
                menu_origin[1] - i * 60 - 32 + menu_animation_frame * 4,
            )

            cv2.circle(frame, (int(x), int(y)), 25, (0, 0, 0), -1)
            cv2.circle(frame, (int(x), int(y)), 20, (255, 255, 255), -1)
            if menu_cursor is not None:
                if dist_between(x, y, menu_cursor[0], menu_cursor[1]) < 30:
                    cv2.circle(frame, (int(x), int(y)), 25, (255, 255, 255), -1)
                    cv2.circle(frame, (int(x), int(y)), 20, (88, 178, 248), -1)

    cv2.imshow("handiwork", frame)

    # handle keyboard input
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the VideoCapture and close all windows
cap.release()
cv2.destroyAllWindows()
