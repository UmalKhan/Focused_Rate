import numpy as np
import cv2
import glob

chessboard = (7, 7)
sqr_size = 8 # mm

images_dir = "calibration_images/"
filename = "camera_calibration.npz"

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2) * sqr_size

objpoints = []
imgpoints = []

print(f"chessboard: {chessboard} inner corners. Square size: {sqr_size}mm.")
print(f"Calibration results will be saved to '{filename}'.")

def capture_calibration_images():
    """Captures calibration images from webcam."""
    print("\n--- Capturing Calibration Images ---")
    print("Hold the chessboard in various orientations and distances.")
    print(f"Press 's' to save an image. Press 'q' to quit capture.")
    print(f"Images will be saved to '{images_dir}'")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return False

    import os
    os.makedirs(images_dir, exist_ok=True)

    img_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        cv2.putText(frame, "Press 's' to Save, 'q' to Quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Saved: {img_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Calibration - Live Feed', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            img_filename = os.path.join(images_dir, f"cal_img_{img_count:03d}.png")
            cv2.imwrite(img_filename, frame)
            print(f"Saved {img_filename}")
            img_count += 1
            cv2.waitKey(500)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished capturing. Captured {img_count} images.")
    return True

def process_calibration_images():
    """Processes images from disk to find chessboard corners."""
    print(f"\n--- Processing Images from {images_dir} ---")
    images = glob.glob(images_dir + '*.png')

    if not images:
        print(f"No images found in '{images_dir}'. Please capture some first or place them there.")
        return False

    processed_count = 0
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"Warning: Could not read image {fname}, skipping.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, chessboard, None)

        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            cv2.drawChessboardCorners(img, chessboard, corners2, ret)
            cv2.imshow('Calibration - Found Corners', img)
            cv2.waitKey(500)
            processed_count += 1
            print(f"Processed: {fname} (Found corners)")
        else:
            print(f"Processed: {fname} (Corners NOT found)")

    cv2.destroyAllWindows()
    print(f"Finished processing images. Found corners in {processed_count} images.")
    return True

if __name__ == '__main__':
    print("Do you want to (c)apture new calibration images or (p)rocess existing ones?")
    choice = input("Enter 'c' or 'p': ").lower()

    if choice == 'c':
        if not capture_calibration_images():
            exit()
        if not process_calibration_images():
            exit()
    elif choice == 'p':
        if not process_calibration_images():
            exit()
    else:
        print("Invalid choice. Exiting.")
        exit()

    if len(objpoints) < 5:
        print(f"\nError: Not enough successful corner detections ({len(objpoints)} found).")
        print("Need at least 5-10 successful detections for reliable calibration.")
        print("Please ensure the chessboard is fully visible, well-lit, and at different angles/distances.")
        exit()

    try:
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )

        if ret:
            print("\nCalibration successful!")
            print("\nCamera Matrix (Intrinsic Parameters):")
            print(camera_matrix)
            print("\nDistortion Coefficients:")
            print(dist_coeffs)

            np.savez(
                filename,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                rvecs=rvecs,
                tvecs=tvecs
            )
            print(f"\nCalibration data saved to '{filename}'")
            print("\nYou can now load these values into your main gaze detection script.")
        else:
            print("\nCalibration failed.")

    except Exception as e:
        print(f"\nAn error occurred during calibration: {e}")
        print("This might happen if the detected corners are not consistent or if there are too few points.")