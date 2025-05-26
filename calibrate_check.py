import cv2
import mediapipe as mp
import numpy as np
import csv

total_time = 1
people_focused = 0

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


# set these values according to your camera
camera_matrix = np.array([
    [640, 0, 320],
    [0, 640, 240],
    [0, 0, 1]
], dtype="double")

dist_coeffs = np.zeros((4, 1))

# MediaPipe landmarks used:
# 33: Left eye inner corner
# 263: Right eye inner corner
# 1: Nose tip
# 61: Left mouth corner
# 291: Right mouth corner
# 199: Chin
model_points = np.array([
    (0.0, 0.0, 0.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
    (0.0, -330.0, -65.0)
], dtype="double")

mediapipe_to_model_idx = {
    1: 0,   # Nose tip
    33: 1,  # Left eye inner corner
    263: 2, # Right eye inner corner
    61: 3,  # Left mouth corner
    291: 4, # Right mouth corner
    199: 5  # Chin
}

# measure these coordinates relative to your camera's origin (which is usually the center of its lens).
TARGET_POINT_3D = np.array([[-500.0, 0.0, 5000.0]])
# cap = cv2.VideoCapture(0)
video_path = 'C:\\Users\\hp\\Desktop\\T_T\\class\\12th Grade ELA.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

print("Ready. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of stream or error reading frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    overall_status_display = "No face detected"

    if results.multi_face_landmarks:
        for face_idx, landmarks_proto in enumerate(results.multi_face_landmarks):
            image_points = []
            for mp_idx, model_idx in mediapipe_to_model_idx.items():
                lm = landmarks_proto.landmark[mp_idx]
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                image_points.append([x, y])
            image_points = np.array(image_points, dtype="double")

            if len(image_points) == len(model_points):
                (success, rotation_vector, translation_vector) = cv2.solvePnP(
                    model_points, image_points, camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

                head_forward_vector = np.array([[0.0, 0.0, 1.0]]).T

                gaze_direction_camera_coords = rotation_matrix @ head_forward_vector
                gaze_direction_camera_coords = gaze_direction_camera_coords.flatten()

                vector_to_target = TARGET_POINT_3D.flatten() - translation_vector.flatten()

                gaze_norm = np.linalg.norm(gaze_direction_camera_coords)
                target_norm = np.linalg.norm(vector_to_target)

                if gaze_norm > 0 and target_norm > 0:
                    dot_product = np.dot(gaze_direction_camera_coords, vector_to_target)
                    cosine_angle = dot_product / (gaze_norm * target_norm)
                    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
                    angle_rad = np.arccos(cosine_angle)
                    angle_deg = np.degrees(angle_rad)

                    GAZE_ANGLE_THRESHOLD_DEG = 10

                    if angle_deg < GAZE_ANGLE_THRESHOLD_DEG:
                        current_face_status = f"Looking at Target ({angle_deg:.1f}°)"
                        people_focused += 1
                    else:
                        if angle_deg < 30:
                             current_face_status = f"Looking Near Target ({angle_deg:.1f}°)"
                        else:
                             current_face_status = f"Looking Away ({angle_deg:.1f}°)"
                else:
                    current_face_status = "Error: Gaze calculation"

                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks_proto,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec
                )

                target_point_2d, _ = cv2.projectPoints(TARGET_POINT_3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                if target_point_2d is not None and len(target_point_2d) > 0:
                    x_target, y_target = int(target_point_2d[0][0][0]), int(target_point_2d[0][0][1])
                    # cv2.circle(frame, (x_target, y_target), 10, (0, 0, 255), -1)

                nose_2d = image_points[0]
                line_end_x = int(nose_2d[0] + gaze_direction_camera_coords[0] * 100)
                line_end_y = int(nose_2d[1] + gaze_direction_camera_coords[1] * 100)
                cv2.arrowedLine(frame, tuple(nose_2d.astype(int)), (line_end_x, line_end_y), (255, 0, 0), 2)

                text_x = int(image_points[0][0])
                text_y = int(image_points[0][1]) - 50
                cv2.putText(frame, f"{current_face_status}",
                            (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

                overall_status_display = "Faces detected"

    # if overall_status_display == "No face detected":
    #     cv2.putText(frame, overall_status_display, (20, 50),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Gaze Target Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    total_time += 1

csv_headers = ['class', 'lecturer', 'course', 'total_time', 'focused_time']
csv_file = open('students.csv', mode='a', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(csv_headers)
csv_writer.writerow(['AB-1A', 'Mr.ABC', 'ABC123', total_time, people_focused])

cap.release()
cv2.destroyAllWindows()
