"""
BINARY COUNTING
-------

Count in binary with your hands! Each finger represents a bit - the pinky represents 2 ** 0, the ring finger represents 2 ** 1, and so on.
"""

import math

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands


hands = mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
)


def finger_is_up(bot, mid_bot, mid_top, top):
    bot_to_mid = math.atan2(mid_bot.y - bot.y, mid_bot.x - bot.x)
    mid_to_tip = math.atan2(top.y - mid_top.y, top.x - mid_top.x)
    return abs(bot_to_mid - mid_to_tip) < math.pi / 2


# Initialize OpenCV
cap = cv2.VideoCapture(2)  # Use 0 for the default camera

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

            total = 0

            for i, points in enumerate(
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
                ]
            ):
                if finger_is_up(*points):
                    total += 2**i

            cv2.putText(
                frame, str(total), (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2
            )

    # Display the resulting frame
    cv2.imshow("handiwork", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the VideoCapture and close all windows
cap.release()
cv2.destroyAllWindows()
