import math

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
HandLandmark = mp.solutions.hands.HandLandmark
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


class EasyHandLandmarker:
    def __init__(self, **kwargs):
        self.tracking_result = None
        self.landmarker = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path="hand_landmarker.task",
                    delegate=BaseOptions.Delegate.GPU,
                ),
                running_mode=VisionRunningMode.LIVE_STREAM,
                result_callback=self._process_result,
                **kwargs,
            )
        )

    def process_frame(self, frame: cv2.typing.MatLike):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.landmarker.detect_async(mp_image, cv2.getTickCount())

    def _process_result(
        self, result: HandLandmarkerResult, _output_image: mp.Image, _timestamp_ms: int
    ):
        self.tracking_result = result

    def get_latest_result(self) -> HandLandmarkerResult:
        return self.tracking_result

    def close(self):
        self.landmarker.close()


def draw_landmarks(frame: cv2.typing.MatLike, result: HandLandmarkerResult):
    hand_landmarks_list = result.hand_landmarks

    for hand_landmarks in hand_landmarks_list:
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in hand_landmarks
            ]
        )
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
        )


def fraction_to_pixels(mat, x, y):
    height, width = mat.shape[:2]
    return x * width, y * height


def pixels_to_fraction(mat, x, y):
    height, width = mat.shape[:2]
    return x / width, y / height


def dist_between_squared(x1, y1, x2, y2):
    return (y2 - y1) ** 2 + (x2 - x1) ** 2


def dist_between(x1, y1, x2, y2):
    return math.hypot(y2 - y1, x2 - x1)


def is_pinch(landmarks):
    index = landmarks[HandLandmark.INDEX_FINGER_TIP]
    thumb = landmarks[HandLandmark.THUMB_TIP]
    wrist = landmarks[HandLandmark.WRIST]

    thumb_to_index = dist_between_squared(thumb.x, thumb.y, index.x, index.y)
    thumb_to_wrist = dist_between_squared(thumb.x, thumb.y, wrist.x, wrist.y)
    index_to_wrist = dist_between_squared(index.x, index.y, wrist.x, wrist.y)

    return (
        thumb_to_index / thumb_to_wrist <= 0.08
        and thumb_to_index / index_to_wrist <= 0.2
    )


def atan2(p1, p2):
    return math.atan2(p2.y - p1.y, p2.x - p1.x)


def get_pinch_pointer(landmarks):
    index = landmarks[HandLandmark.INDEX_FINGER_TIP]
    thumb = landmarks[HandLandmark.THUMB_TIP]
    return (index.x + thumb.x) / 2, (index.y + thumb.y) / 2


def finger_is_up(bot, mid_bot, mid_top, top):
    bot_to_mid = atan2(bot, mid_bot)
    mid_to_tip = atan2(mid_top, top)
    return abs(bot_to_mid - mid_to_tip) < math.pi / 4


def fingers_are_up(landmarks):
    return list(
        map(
            lambda finger: finger_is_up(*finger),
            [
                landmarks[HandLandmark.THUMB_CMC : HandLandmark.THUMB_TIP + 1],
                landmarks[
                    HandLandmark.INDEX_FINGER_MCP : HandLandmark.INDEX_FINGER_TIP + 1
                ],
                landmarks[
                    HandLandmark.MIDDLE_FINGER_MCP : HandLandmark.MIDDLE_FINGER_TIP + 1
                ],
                landmarks[
                    HandLandmark.RING_FINGER_MCP : HandLandmark.RING_FINGER_TIP + 1
                ],
                landmarks[HandLandmark.PINKY_MCP : HandLandmark.PINKY_TIP + 1],
            ],
        )
    )
