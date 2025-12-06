import cv2
import numpy as np


def intersection(l1, l2, img_shape):
    h, w = img_shape

    rho1, theta1 = l1
    rho2, theta2 = l2

    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([rho1, rho2])

    x, y = np.linalg.solve(A, b)
    x, y = int(round(x)), int(round(y))

    if not (0 <= x < w and 0 <= y < h):
        return None

    return [x, y]


def order_points(pts):
    pts = np.array(pts)
    ordered = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()

    ordered[0] = pts[np.argmin(s)]   # top left
    ordered[2] = pts[np.argmax(s)]   # bottom right
    ordered[1] = pts[np.argmin(diff)]  # top right
    ordered[3] = pts[np.argmax(diff)]  # bottom left

    return ordered

def find_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)

        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    if lines is not None:
        horiz, vert = [], []

        for rho, theta in lines[:, 0]:
            angle = theta * 180 / np.pi
            if angle < 10 or angle > 170:
                vert.append((rho, theta))
            elif 80 < angle < 100:
                horiz.append((rho, theta))

        horiz = horiz[:2]
        vert = vert[:2]

        corners = [
            intersection(h, v, gray.shape)
            for h in horiz for v in vert
        ]
        corners = [c for c in corners if c]

        if len(corners) >= 4:
            return np.array(corners[:4], dtype=np.float32)

    h, w = img.shape[:2]
    return np.array([
        [50, 50],
        [w - 50, 50],
        [w - 50, h - 50],
        [50, h - 50]
    ], dtype=np.float32)

def auto_warp_image(input_path, output_path, out_size):
    img = cv2.imread(input_path)
    corners = find_corners(img)
    ordered = order_points(corners)

    w, h = out_size
    destination = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered, destination)
    warped = cv2.warpPerspective(img, M, (w, h))
    cv2.imwrite(output_path, warped)
    return warped


images = [
    ("images/quote.jpg","images/quote_lines_warped.jpg",(400, 300)),
    ("images/quote2.jpg","images/quote2_lines_warped.jpg",(400, 300)),
    ("images/sudoku.jpg","images/sudoku_lines_warped.jpg",(500, 500)),
    ("images/program_sheet.jpg", "images/program_sheet_lines_warped.jpg",(500, 600)),
]

for src, dst, size in images:
    warped = auto_warp_image(src, dst, size)
    if warped is not None:
        cv2.imshow(src, warped)

while cv2.waitKey(1) != 27:
    pass

cv2.destroyAllWindows()
