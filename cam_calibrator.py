import os
import numpy as np
import cv2
import glob

cam_name = "realsense_d435"
img_dir = f"{cam_name}_imgs"
square_length = 2.4
board_shape = 7, 10
n_corners = n_corners_y, n_corners_x = board_shape[0] - 1, board_shape[1] - 1
image_index = 0
alpha = 1


def get_folder_size(dir):
    return len(os.listdir(dir))


def get_test_image():
    directory = os.getcwd()

    if not directory.endswith("cam_calibration"):
        raise ImportError("Current directory is not named 'cam_calibration'. Script is most likely run from wrong root.")

    if not img_dir in os.listdir():
        raise ImportError(f"{img_dir} does not exist.")

    if not get_folder_size(img_dir) > 0:
        raise ImportError(f"{img_dir} is empty")

    print(f"{img_dir}/{cam_name}_capture_0")
    test_img = cv2.imread(f"{img_dir}/{cam_name}_capture_0.png")
    return test_img


world_frames = []
camera_frames = []

termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
sub_winsize = 11, 11
sub_zerozone = -1, -1

object_points_3D = np.zeros((n_corners_y * n_corners_x, 3), np.float32)
object_points_3D[:,:2] = np.mgrid[:n_corners_y, :n_corners_x].T.reshape(-1, 2)
object_points_3D *= square_length

def get_calibration_matrix(display=False):
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

    test_img = get_test_image()
    img_size = test_img.shape[:2]
    
    for image_file in os.listdir(img_dir):

        image = cv2.imread(f"{img_dir}/{image_file}")
        img_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(img_grayscale, n_corners, None)
        if ret:
            world_frames.append(object_points_3D)
            corners_subpixels = cv2.cornerSubPix(img_grayscale, corners, sub_winsize, sub_zerozone, termination_criteria)
            camera_frames.append(corners_subpixels)
            
            cv2.drawChessboardCorners(image, n_corners, corners_subpixels, ret)

            if display:
                cv2.imshow("Image", image)
                cv2.waitKey(200)

    _, camera_matrix, distortion, rotation_vectors, translation_vectors = cv2.calibrateCamera(world_frames, camera_frames, img_grayscale.shape[::-1], None, None)
    optimal_camera_matrix, region = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion, img_size, alpha, img_size)

    return camera_matrix, distortion, optimal_camera_matrix


def main():
    camera_matrix, distortion, optimal_camera_matrix = get_calibration_matrix()
    print(camera_matrix)


if __name__ == "__main__":
    main()