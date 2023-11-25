import math

import mediapipe as mp

mp_hands = mp.solutions.hands


def fraction_to_pixels(mat, x, y):
    height, width = mat.shape[:2]
    return x * width, y * height


def point_average(*points):
    return (
        sum(point.x for point in points) / len(points),
        sum(point.y for point in points) / len(points),
    )


def dist_between(x1, y1, x2, y2):
    return math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)


def is_pinch(landmarks):
    index = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    return dist_between(thumb.x, thumb.y, index.x, index.y) <= 0.04


def get_pinch_pointer(landmarks):
    index = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    return point_average(index, thumb)
