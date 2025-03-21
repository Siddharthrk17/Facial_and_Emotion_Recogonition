import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace

# Initialize MediaPipe Face Detection and Face Mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB (MediaPipe requires RGB format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape

            # Convert bbox to pixel values
            x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            x, y = max(x, 0), max(y, 0)

            # Extract face ROI
            face_roi = frame[y:y + h_box, x:x + w_box]

            # Perform emotion detection with DeepFace
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion'] if result else "Unknown"
            except:
                emotion = "Unknown"

            # Process the face with MediaPipe Face Mesh
            face_landmarks = face_mesh.process(rgb_frame)

            if face_landmarks.multi_face_landmarks:
                for landmarks in face_landmarks.multi_face_landmarks:
                    # Extract key facial points
                    eye_left = (int(landmarks.landmark[33].x * w), int(landmarks.landmark[33].y * h))  # Left eye
                    eye_right = (int(landmarks.landmark[263].x * w), int(landmarks.landmark[263].y * h))  # Right eye
                    nose = (int(landmarks.landmark[1].x * w), int(landmarks.landmark[1].y * h))  # Nose tip
                    mouth_left = (int(landmarks.landmark[61].x * w), int(landmarks.landmark[61].y * h))  # Left mouth corner
                    mouth_right = (int(landmarks.landmark[291].x * w), int(landmarks.landmark[291].y * h))  # Right mouth corner

                    # Draw facial landmarks
                    cv2.circle(frame, eye_left, 3, (0, 255, 255), -1)
                    cv2.circle(frame, eye_right, 3, (0, 255, 255), -1)
                    cv2.circle(frame, nose, 3, (255, 0, 255), -1)
                    cv2.circle(frame, mouth_left, 3, (255, 0, 0), -1)
                    cv2.circle(frame, mouth_right, 3, (255, 0, 0), -1)

            # Draw bounding box and emotion text
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    else:
        # If no face is detected, show message
        cv2.putText(frame, "Face Not Found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Real-time Emotion Detection with MediaPipe', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
