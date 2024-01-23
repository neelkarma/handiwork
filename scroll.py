"""
FRICTION SCROLLING
----

Pinch and drag to scroll. Velocity is preserved on release, and decreases over time after release, like on a smartphone.

Press 'r' to reset scroll position.
"""

import textwrap

import cv2
import mediapipe as mp

from common.hands import fraction_to_pixels, get_pinch_pointer, is_pinch

mp_hands = mp.solutions.hands

FRICTION_COEFF = 0.15

hands = mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
)


# Initialize OpenCV
cap = cv2.VideoCapture(2)  # Use 0 for the default camera

old_pos = None
last_pos = None
current_pos = 0
velocity = None
is_scrolling = False
scroll_origin = None
lorem = textwrap.wrap(
    "\n".join(
        [
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse sed orci non eros tempor bibendum. Mauris mauris mauris, tempor non dignissim ut, placerat ac ex. Donec eu ornare dolor. Morbi aliquam gravida condimentum. Ut a turpis sit amet lectus dignissim congue eget dictum dui. Nam ex ipsum, dapibus id congue nec, ullamcorper nec lacus. Etiam in felis accumsan, volutpat felis ac, pretium nisi. Integer et neque ut tortor mollis finibus. Donec lacinia porta arcu, vitae pharetra nibh semper sit amet. Sed ac lobortis ligula, ut pretium enim. Vivamus ut ante metus. Cras libero arcu, imperdiet ut rutrum quis, laoreet ac turpis.",
            "In tristique id ante eget interdum. Quisque rutrum lectus vel feugiat eleifend. Praesent nec pellentesque nulla. Aliquam nec erat in purus rhoncus varius eu auctor nunc. In maximus mollis ante, sit amet maximus urna congue vitae. Maecenas vitae malesuada nisl. In porttitor et lorem sed dignissim. Donec ac est interdum nisl rutrum maximus. Morbi eget vestibulum elit. Phasellus rutrum ex non lorem euismod, sed tempor felis volutpat. Interdum et malesuada fames ac ante ipsum primis in faucibus. Integer sagittis eget magna facilisis venenatis. Nullam ullamcorper condimentum nibh, a vehicula metus sollicitudin a.",
            "Aliquam et placerat diam, id ultricies magna. Duis efficitur eleifend feugiat. Aliquam sed orci quis odio tincidunt fermentum at vitae tellus. Suspendisse tortor nunc, luctus sed euismod sit amet, condimentum in risus. Nam euismod in nulla vitae egestas. Vivamus et risus libero. Ut efficitur est eu metus sollicitudin, sit amet cursus lorem egestas. Pellentesque nulla ipsum, volutpat ac eros a, vehicula eleifend dui. Donec hendrerit, mi sit amet iaculis porttitor, odio turpis gravida mauris, sed bibendum nisi ante non dui. Sed eu metus nec leo aliquam pellentesque.",
            "Aenean facilisis porttitor odio, eu consequat est bibendum at. Duis ultricies ultricies venenatis. Vestibulum molestie metus eget feugiat luctus. Sed ac mauris sapien. In erat ipsum, volutpat in justo sit amet, tempus vulputate massa. Donec non arcu vulputate, egestas ipsum quis, blandit tellus. Nulla augue felis, varius id porttitor et, pharetra tincidunt erat. Pellentesque nisl nibh, bibendum ac laoreet eu, mollis vitae odio.",
            "Duis facilisis laoreet libero sit amet iaculis. Integer erat libero, condimentum at sodales id, sollicitudin id ligula. Aliquam fermentum ligula et fringilla feugiat. Ut et tristique tellus. Phasellus mi orci, rhoncus non nisl sit amet, auctor volutpat metus. Etiam laoreet libero eu velit luctus, eget vulputate metus efficitur. Cras congue vel ante at mattis. Nullam in semper felis, sed iaculis nulla. Ut ac tellus dolor. Mauris sed elit tincidunt, placerat enim ac, hendrerit ligula. Sed quis mauris et eros elementum venenatis.",
        ]
    ),
    30,
)

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

            pinching = is_pinch(landmarks)

            if pinching and not is_scrolling:  # is scrolling
                is_scrolling = True
                old_pos = current_pos
                scroll_origin = fraction_to_pixels(
                    frame, *get_pinch_pointer(landmarks)
                )[1]
            elif pinching and is_scrolling:
                last_pos = current_pos
                current_pos = old_pos + (
                    fraction_to_pixels(frame, *get_pinch_pointer(landmarks))[1]
                    - scroll_origin
                )
            elif not pinching and is_scrolling:
                is_scrolling = False
                if last_pos is not None:
                    velocity = current_pos - last_pos
                old_pos = None
                last_pos = None
                scroll_origin = None
    else:
        if is_scrolling:
            is_scrolling = False
            if last_pos is not None:
                velocity = current_pos - last_pos
            old_pos = None
            last_pos = None
            scroll_origin = None

    if not is_scrolling and velocity is not None:
        current_pos += velocity
        velocity *= 1 - FRICTION_COEFF

    for i, line in enumerate(lorem):
        cv2.putText(
            frame,
            line,
            (0, int(current_pos) + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255) if is_scrolling else (255, 0, 0),
            2,
        )

    # Display the resulting frame
    cv2.imshow("handiwork", frame)

    # handle keyboard input
    key = cv2.waitKey(1)
    # reset if 'r' is pressed
    if key & 0xFF == ord("r"):
        current_pos = 0
    # Break the loop if 'q' is pressed
    if key & 0xFF == ord("q"):
        break

# Release the VideoCapture and close all windows
cap.release()
cv2.destroyAllWindows()
