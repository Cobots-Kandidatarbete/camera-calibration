import cv2


cam_name = "realsense_d435"
img_dir = f"{cam_name}_imgs"
marked_dir = f"{cam_name}_marked"
square_length = 2.4
board_shape = 7, 10
n_inner_corners = board_shape[0] - 1, board_shape[1] - 1


def draw_corners(img_index):
    img_name = f"{cam_name}_capture_{img_index}"
    img_path = f"{img_dir}/{img_path}"
    image = cv2.imread(f"{img_path}.png")
    
    img_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(img_grayscale, n_inner_corners, None)

    if ret:
        marked_path = f"{marked_dir}/{img_name}"
        cv2.drawChessboardCorners(image, n_inner_corners, corners, ret)
        cv2.imwrite(marked_path, image)
        cv2.imshow("Image", image)

        cv2.waitKey(0)

        cv2.destroyAllWindows()
        

if __name__ == "__main__":
    draw_corners()