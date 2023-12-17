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


def finger_is_up(bot, mid_bot, mid_top, top):
    bot_to_mid = math.atan2(mid_bot.y - bot.y, mid_bot.x - bot.x)
    mid_to_tip = math.atan2(top.y - mid_top.y, top.x - mid_top.x)
    return abs(bot_to_mid - mid_to_tip) < math.pi / 4


def fingers_are_up(landmarks):
    return list(
        map(
            lambda finger: finger_is_up(*finger),
            [
                landmarks.landmark[
                    mp_hands.HandLandmark.THUMB_CMC : mp_hands.HandLandmark.THUMB_TIP
                    + 1
                ],
                landmarks.landmark[
                    mp_hands.HandLandmark.INDEX_FINGER_MCP : mp_hands.HandLandmark.INDEX_FINGER_TIP
                    + 1
                ],
                landmarks.landmark[
                    mp_hands.HandLandmark.MIDDLE_FINGER_MCP : mp_hands.HandLandmark.MIDDLE_FINGER_TIP
                    + 1
                ],
                landmarks.landmark[
                    mp_hands.HandLandmark.RING_FINGER_MCP : mp_hands.HandLandmark.RING_FINGER_TIP
                    + 1
                ],
                landmarks.landmark[
                    mp_hands.HandLandmark.PINKY_MCP : mp_hands.HandLandmark.PINKY_TIP
                    + 1
                ],
            ],
        )
    )
