import os
import numpy as np
import cv2


cam_name = "realsense_d435"
img_dir = f"{cam_name}_imgs"
config_dir = f"{cam_name}_configs"
config_name = f"{cam_name}_config"
square_length = 2.4
board_shape = 7, 10
n_corners = n_corners_y, n_corners_x = board_shape[0] - 1, board_shape[1] - 1
image_index = 0
alpha = 1


def get_folder_size(dir):
    return len(os.listdir(dir))


def get_test_image():
    directory = os.getcwd()

    if not directory.endswith("camera-calibration"):
        raise ImportError("Current directory is not named 'camera-calibration'. Script is most likely run from wrong root.")

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

def get_calibration_matrix(display=False, undistort_test=False, save_file=False):
    cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)

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
                cv2.waitKey(1000)

    _, camera_matrix, distortion_matrix, rotation_vectors, translation_vectors = cv2.calibrateCamera(world_frames, camera_frames, img_grayscale.shape[::-1], None, None)
    optimal_camera_matrix, region = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_matrix, img_size[::-1], alpha, img_size[::-1])
    print(img_size[::-1])

    if undistort_test:
        undistorted_test_img = cv2.undistort(test_img, camera_matrix, distortion_matrix, None, optimal_camera_matrix)

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

        while True:
            key_input = cv2.waitKey(0 & 0xFF)
            if key_input == ord('1'):
                cv2.imshow("Image", test_img)
            elif key_input == ord('2'):
                cv2.imshow("Image", undistorted_test_img)
            elif key_input == ord('q'):
                break


    if save_file:
        if config_dir not in os.listdir():
            os.mkdir(config_dir)
            print(f"Created {config_dir}")

        config_path = f"{config_dir}/{config_name}.yaml"
        config_file = cv2.FileStorage(config_path, cv2.FILE_STORAGE_WRITE)
        config_file.write('camera_matrix', camera_matrix)
        config_file.write('distortion_matrix', distortion_matrix)
        config_file.write('optimal_camera_matrix', optimal_camera_matrix)
        config_file.write('image_size', img_size[::-1])
        config_file.release()


    cv2.destroyAllWindows()

    return camera_matrix, distortion_matrix, optimal_camera_matrix


def main():
    camera_matrix, distortion, optimal_camera_matrix = get_calibration_matrix(display=False, undistort_test=False, save_file=True)
    print("Camera matrix:")
    print(camera_matrix)
    print("Optimal camera matrix:")
    print(optimal_camera_matrix)


if __name__ == "__main__":
    main()