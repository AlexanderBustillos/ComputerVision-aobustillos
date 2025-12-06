import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
cv2.namedWindow('ArUco Markers', cv2.WINDOW_NORMAL)

for marker_id in range(4):
    marker_image = np.zeros((200, 200), dtype=np.uint8)
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 200, marker_image, 1)
    cv2.imshow('ArUco Markers', marker_image)
    cv2.waitKey(0)

for marker_id in range(4):
    marker_image = np.zeros((200, 200), dtype=np.uint8)
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 200, marker_image, 1)
    filename = f"marker_{marker_id}.png"
    cv2.imwrite(filename, marker_image)


cv2.destroyAllWindows()