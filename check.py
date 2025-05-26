import cv2
import mediapipe as mp
import math
import csv

people_focused = 0
total_time = 1

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=20,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# cap = cv2.VideoCapture(0)
video_path = 'C:\\Users\\hp\\Desktop\\T_T\\class\\12th Grade ELA.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened successfully. Press 'q' to quit.")

down_threshold = 0.06
up_threshold = 0.05
horizontal_threshold = 0.03

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    current_status = " "

    if results.multi_face_landmarks:
        for face_idx, landmarks in enumerate(results.multi_face_landmarks):

        # landmarks = results.multi_face_landmarks[0].landmark

            # mp_drawing.draw_landmarks(
            #     image=frame,
            #     landmark_list=results.multi_face_landmarks[0],
            #     connections=mp_face_mesh.FACEMESH_TESSELATION,
            #     landmark_drawing_spec=drawing_spec,
            #     connection_drawing_spec=drawing_spec
            # )

            # landmark indices (approximate for simplicity):
            # Nose tip: 1
            # Left eye inner corner: 33
            # Right eye inner corner: 263
            # Chin: 199 (or a lower face point)

            img_h, img_w, _ = frame.shape
            nose_tip = landmarks.landmark[1]
            left_eye_inner = landmarks.landmark[33]
            right_eye_inner = landmarks.landmark[263]
            chin = landmarks.landmark[199]

            face_center_x = (left_eye_inner.x + right_eye_inner.x) / 2
            avg_eye_y = (left_eye_inner.y + right_eye_inner.y) / 2

            current_status = "Focused"

            if nose_tip.y - avg_eye_y > down_threshold:
                current_status = "Not Focused (down)"
            elif avg_eye_y - nose_tip.y > up_threshold:
                current_status = "Not Focused (up)"

            if current_status == "Focused":
                if nose_tip.x < face_center_x - horizontal_threshold:
                    current_status = "Not Focused (left)"
                elif nose_tip.x > face_center_x + horizontal_threshold:
                    current_status = "Not Focused (right)"
                else:
                    people_focused += 1

            img_h, img_w, _ = frame.shape
            text_x = int(nose_tip.x * img_w)
            text_y = int(nose_tip.y * img_h) - 30

            cv2.putText(frame, f"{current_status}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, current_status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('Face Focus Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(people_focused/total_time)
        break

    total_time += 1

csv_headers = ['class', 'lecturer', 'course', 'total_time', 'focused_time']
csv_file = open('students.csv', mode='a', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(csv_headers)
csv_writer.writerow(['AB-1A', 'Mr.ABC', 'ABC123', total_time, people_focused])

cap.release()
cv2.destroyAllWindows()
