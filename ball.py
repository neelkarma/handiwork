"""
BALL
------

Play with the ball! Pinch to hold, drag and then release to throw.

Tip: Modify the NUM_BALLS constant below :)
"""

import random

import cv2
import mediapipe as mp

from common.hands import (
    dist_between_squared,
    fraction_to_pixels,
    get_pinch_pointer,
    is_pinch,
)

mp_hands = mp.solutions.hands

GRAVITY = 3
BOUNCE_DAMPING = 0.3
HORIZONTAL_RESISTANCE = 0.05
RELEASE_SENSITIVITY = 1
BALL_RADIUS = 40
NUM_BALLS = 1


hands = mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
)


class Ball:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r
        self.vy = 0
        self.vx = 20
        self.is_grabbing = False
        self.grab_delta_x = 0
        self.grab_delta_y = 0
        self.last_grab_x = 0
        self.last_grab_y = 0

    def handle_hand(self, handmarks):
        is_pinching = is_pinch(landmarks)
        pointer = get_pinch_pointer(handmarks)
        px, py = fraction_to_pixels(frame, *pointer)
        if is_pinching and not self.is_grabbing:
            self.is_grabbing = (
                dist_between_squared(px, py, self.x, self.y) <= self.r**2
            )

            if self.is_grabbing:
                self.last_grab_x = px
                self.last_grab_y = py
                self.grab_delta_x = self.x - px
                self.grab_delta_y = self.y - py

        elif is_pinching and self.is_grabbing:
            self.x = px + self.grab_delta_x
            self.y = py + self.grab_delta_y
            self.last_grab_x = px
            self.last_grab_y = py
        elif self.is_grabbing:
            self.vx = px - self.last_grab_x
            self.vy = py - self.last_grab_y
            self.is_grabbing = False

    def update(self, frame):
        if self.is_grabbing:
            return

        screen_height, screen_width, _ = frame.shape
        self.vy += GRAVITY
        self.vx -= self.vx * HORIZONTAL_RESISTANCE

        if self.x - self.r + self.vx < 0 or self.x + self.r + self.vx > screen_width:
            self.vx *= -1
        if self.y + self.r + self.vy > screen_height:
            self.vy *= -1 * (1 - BOUNCE_DAMPING)
        if self.y - self.r + self.vy < 0:
            self.vy *= -1

        self.y += self.vy
        self.x += self.vx

    def render(self, frame):
        frame = cv2.circle(
            frame,
            (int(self.x), int(self.y)),
            self.r,
            (0, 255, 0) if self.is_grabbing else (0, 0, 255),
            -1,
        )
        frame = cv2.circle(
            frame,
            (int(self.x), int(self.y)),
            self.r,
            (0, 0, 0),
            2,
        )
        return frame


# Initialize OpenCV
cap = cv2.VideoCapture(2)  # Use 0 for the default camera
balls = [
    Ball(
        random.randint(BALL_RADIUS, 500), random.randint(BALL_RADIUS, 500), BALL_RADIUS
    )
    for _ in range(NUM_BALLS)
]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    for ball in balls:
        frame = ball.render(frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the image
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            for ball in balls:
                ball.handle_hand(landmarks)

    for ball in balls:
        ball.update(frame)

    # Display the resulting frame
    cv2.imshow("handiwork", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the VideoCapture and close all windows
cap.release()
cv2.destroyAllWindows()
