import cv2
import mediapipe as mp

from common import fraction_to_pixels, get_pinch_pointer, is_pinch

mp_hands = mp.solutions.hands


hands = mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
)

# Initialize OpenCV
cap = cv2.VideoCapture(2)  # Use 0 for the default camera
was_drawing = False
drawing = [[]]

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
                was_drawing = True
                pointer = get_pinch_pointer(landmarks)
                pointer = fraction_to_pixels(frame, pointer[0], pointer[1])
                drawing[-1].append((int(pointer[0]), int(pointer[1])))
            elif was_drawing:
                was_drawing = False
                drawing.append([])

    for stroke in drawing:
        for i in range(1, len(stroke)):
            x1, y1 = stroke[i - 1]
            x2, y2 = stroke[i]
            image = cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)

    # Display the resulting frame
    cv2.imshow("handiwork", frame)

    key = cv2.waitKey(1)
    # reset canvas is 'r' is pressed
    if key & 0xFF == ord("r"):
        drawing = [[]]

    # Break the loop if 'q' is pressed
    if key & 0xFF == ord("q"):
        break

# Release the VideoCapture and close all windows
cap.release()
cv2.destroyAllWindows()
