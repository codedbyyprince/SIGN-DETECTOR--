import cv2
import mediapipe as mp
import os
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Directory to save data
DATA_DIR = "sign_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Function to save landmarks
def save_landmarks(landmarks, label):
    file_path = os.path.join(DATA_DIR, f"{label}.csv")
    with open(file_path, "a") as f:
        f.write(",".join(map(str, landmarks)) + "\n")

# Start webcam feed
cap = cv2.VideoCapture(0)
fps_time = 0  # For FPS calculation
frame_skip = 2  # Process every nth frame
frame_count = 0

print("Press 'q' to quit. Enter the label for the sign when prompted.")

label = input("Enter the label for the sign to capture: ")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Skip frames to reduce processing
    if frame_count % frame_skip != 0:
        continue

    frame = cv2.flip(frame, 1)  # Flip for mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Save landmarks
            save_landmarks(landmarks, label)

    # Display the frame with FPS
    fps = int(1 / (time.time() - fps_time + 1e-6))
    fps_time = time.time()
    cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Label: {label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Sign Data Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()