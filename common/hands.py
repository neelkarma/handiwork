import math

import mediapipe as mp

mp_hands = mp.solutions.hands


def fraction_to_pixels(mat, x, y):
    height, width = mat.shape[:2]
    return x * width, y * height


def pixels_to_fraction(mat, x, y):
    height, width = mat.shape[:2]
    return x / width, y / height


def dist_between_squared(x1, y1, x2, y2):
    return (y2 - y1) ** 2 + (x2 - x1) ** 2


def dist_between(x1, y1, x2, y2):
    return math.sqrt(dist_between_squared(x1, y1, x2, y2))


def is_pinch(landmarks):
    index = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    thumb_to_index = dist_between_squared(thumb.x, thumb.y, index.x, index.y)
    thumb_to_wrist = dist_between_squared(thumb.x, thumb.y, wrist.x, wrist.y)
    thumb_to_middle = dist_between_squared(thumb.x, thumb.y, middle.x, middle.y)

    return (
        thumb_to_index / thumb_to_wrist <= 0.05
        and thumb_to_index / thumb_to_middle <= 0.3
    )


def get_pinch_pointer(landmarks):
    index = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    return (index.x + thumb.x) / 2, (index.y + thumb.y) / 2
