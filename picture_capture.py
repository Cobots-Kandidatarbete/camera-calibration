import os
import cv2


active_cam = 6
cam_name = "realsense_d435"
img_dir = f"{cam_name}_imgs"

def main():
    directory = os.getcwd()

    if not directory.endswith("cam_calibration"):
        raise ImportError("Current directory is not named 'cam_calibration'. Script is most likely run from wrong root.")

    if not img_dir in os.listdir():
        os.mkdir(img_dir)
        print(f"Created folder: {img_dir}")

    run_video_capturing()


def run_video_capturing():

    capture = cv2.VideoCapture(active_cam)
    print("Starting capturing")
    print("Press 'q' or ESC to exit program.")
    print("Press SPACE to capture image.")

    while capture.isOpened():
        ret, frame = capture.read()
        if ret:
            cv2.imshow("Frame", frame)

            key_input = cv2.waitKey(25) & 0xFF

            if key_input == 27 or key_input == ord('q'):
                break
            
            if key_input == ord(' '):
                cv2.imwrite(f"{img_dir}/{cam_name}_capture_{get_folder_size(img_dir)}.png", frame)
                

        else:
            break


def get_folder_size(dir):
    return len(os.listdir(dir))


if __name__ == "__main__":
    main()