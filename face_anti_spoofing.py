import cv2
import dlib
import numpy as np
import time

# Load the pre-trained face detector and facial landmark predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Pastikan jalur ke file model benar

# Function to calculate eye aspect ratio (EAR)
def calculate_ear(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate mouth aspect ratio (MAR)
def calculate_mar(mouth_points):
    A = np.linalg.norm(mouth_points[2] - mouth_points[10])
    B = np.linalg.norm(mouth_points[4] - mouth_points[8])
    C = np.linalg.norm(mouth_points[0] - mouth_points[6])
    mar = (A + B) / (2.0 * C)
    return mar

# Thresholds and consecutive frame count for blink, mouth open, and head turn detection
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.7
TURN_THRESHOLD = 0.2  # Adjusted threshold to be more sensitive
CONSECUTIVE_FRAMES = 3
COUNTDOWN_DURATION = 4  # Duration in seconds for the final step
blink_counter = 0
mouth_open_counter = 0
head_turn_left_counter = 0
head_turn_right_counter = 0
head_turn_left_done = False
head_turn_right_done = False

# States for the anti-spoofing steps
steps = ['Blink', 'Open Mouth', 'Turn Head Left and Right', 'Face Forward']
current_step = 0
step_completed = False
face_forward_start_time = None

# Initialize video capture
cap = cv2.VideoCapture(0)

# Variables to track head position
initial_nose_position = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if current_step < 3:
        for face in faces:
            landmarks = predictor(gray, face)

            # Extract eye landmarks for EAR calculation
            left_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
            right_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])

            # Extract mouth landmarks for MAR calculation
            mouth_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)])

            # Calculate EAR and MAR
            left_ear = calculate_ear(left_eye_points)
            right_ear = calculate_ear(right_eye_points)
            ear = (left_ear + right_ear) / 2.0
            mar = calculate_mar(mouth_points)

            # Calculate nose position
            nose_point = np.array((landmarks.part(30).x, landmarks.part(30).y))
            left_eye_center = np.mean(left_eye_points, axis=0)
            right_eye_center = np.mean(right_eye_points, axis=0)

            # Calculate the distance from nose to left and right eyes
            dist_nose_left_eye = np.linalg.norm(nose_point - left_eye_center)
            dist_nose_right_eye = np.linalg.norm(nose_point - right_eye_center)

            # Ratio to determine head turn
            turn_ratio = dist_nose_left_eye / dist_nose_right_eye

            # Check current step
            if current_step == 0:  # Blink detection
                if ear < EAR_THRESHOLD:
                    blink_counter += 1
                else:
                    if blink_counter >= CONSECUTIVE_FRAMES:
                        step_completed = True
                    blink_counter = 0

            elif current_step == 1:  # Mouth open detection
                if mar > MAR_THRESHOLD:
                    mouth_open_counter += 1
                else:
                    if mouth_open_counter >= CONSECUTIVE_FRAMES:
                        step_completed = True
                    mouth_open_counter = 0

            elif current_step == 2:  # Head turn detection (left-right)
                if not head_turn_left_done:
                    if turn_ratio < (1 - TURN_THRESHOLD):  # Head turned to the left
                        head_turn_left_counter += 1
                        cv2.putText(frame, "Turn Head Left - OK", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        if head_turn_left_counter >= CONSECUTIVE_FRAMES:
                            head_turn_left_done = True
                            head_turn_left_counter = 0
                        else:
                            cv2.putText(frame, "Turn Head Left", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                elif not head_turn_right_done:
                    if turn_ratio > (1 + TURN_THRESHOLD):  # Head turned to the right
                        head_turn_right_counter += 1
                        cv2.putText(frame, "Turn Head Right - OK", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        if head_turn_right_counter >= CONSECUTIVE_FRAMES:
                            head_turn_right_done = True
                            head_turn_right_counter = 0
                        else:
                            cv2.putText(frame, "Turn Head Right", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if head_turn_left_done and head_turn_right_done:
                    step_completed = True

            # Draw facial landmarks for eyes, mouth, and face
            for n in range(68):
                x, y = landmarks.part(n).x, landmarks.part(n).y
                color = (255, 0, 0)  # Default color (blue) for face landmarks
                if 36 <= n < 42 or 42 <= n < 48:  # Eyes landmarks
                    color = (0, 255, 0)  # Green for eyes landmarks
                elif 48 <= n < 68:  # Mouth landmarks
                    color = (0, 0, 255)  # Red for mouth landmarks
                cv2.circle(frame, (x, y), 2, color, -1)

        if step_completed:
            current_step += 1
            step_completed = False
            if current_step == 3:  # Start the face forward step
                face_forward_start_time = time.time()
                initial_nose_position = nose_point
            elif current_step >= len(steps):
                print("Wajah asli terdeteksi!")
                cap.release()
                cv2.destroyAllWindows()
                exit()

    else:  # Face forward step
        if face_forward_start_time:
            elapsed_time = time.time() - face_forward_start_time
            countdown = int(COUNTDOWN_DURATION - elapsed_time)

            for face in faces:
                landmarks = predictor(gray, face)
                nose_point = np.array((landmarks.part(30).x, landmarks.part(30).y))

                # Check if head is still facing forward and in the same position
                if abs(turn_ratio - 1) > TURN_THRESHOLD * 1.5 or np.linalg.norm(nose_point - initial_nose_position) > 15:
                    # Head is not facing forward or has moved significantly, reset countdown
                    face_forward_start_time = time.time()
                    cv2.putText(frame, "Please face forward", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    # Head is facing forward and in the same position
                    if countdown > 0:
                        cv2.putText(frame, f"Face Forward - {countdown}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        print("Wajah asli terdeteksi!")
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()
        else:
            face_forward_start_time = time.time()  # Initialize timer if not already

    # Draw instructions
    if current_step < 3:
        cv2.putText(frame, f"Step: {steps[current_step]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Face Anti-Spoofing', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
