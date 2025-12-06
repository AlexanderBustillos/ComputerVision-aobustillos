import cv2
import numpy as np

OVERLAY_IMAGE = "jackie.jpg"
INPUT_IMAGES = ["Code0.jpg", "Code1.jpg", "Code2.jpg", "Code3.jpg"]
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
detector = cv2.aruco.ArucoDetector(aruco_dict)


def overlay_on_markers(base_img, overlay_img):
    corners, ids, _ = detector.detectMarkers(base_img)
    if ids is None or len(ids) < 4:
        return None

    ids = ids.flatten()
    corners_dict = {}

    for corner, marker_id in zip(corners, ids):
        if marker_id in [0, 1, 2, 3]:
            corners_dict[marker_id] = corner.reshape(4, 2)

    if len(corners_dict) < 4:
        return None

    dst_pts = np.array([
        corners_dict[0][0],  # top-left
        corners_dict[1][1],  # top-right
        corners_dict[2][2],  # bottom-right
        corners_dict[3][3],  # bottom-left
    ], dtype=np.float32)

    h, w = overlay_img.shape[:2]
    src_pts = np.array([
        [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src_pts, dst_pts)
    if H is None:
        return None

    warped = cv2.warpPerspective(overlay_img, H, (base_img.shape[1], base_img.shape[0]))

    mask = np.any(warped > 0, axis=2)
    result = base_img.copy()
    result[mask] = warped[mask]

    return result

overlay = cv2.imread(OVERLAY_IMAGE)


for i, img_file in enumerate(INPUT_IMAGES):
    base = cv2.imread(img_file)
    result = overlay_on_markers(base, overlay)
    cv2.imwrite(f"overlay_result{i}.jpg", result)
