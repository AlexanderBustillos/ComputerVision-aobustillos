import cv2
import numpy as np


def sign_lines(img: np.ndarray) -> np.ndarray:
    """
    This function takes in the image as a numpy array and returns a numpy array of lines.

    https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
    :param img: Image as numpy array
    :return: Numpy array of lines.
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray,(5,5),0)
    img_edge = cv2.Canny(img_blur, 100, 200)
    img_lines = cv2.HoughLinesP(img_edge, 1, np.pi / 180, 50, None, 50, 10)
    return img_lines


def sign_circle(img: np.ndarray) -> np.ndarray:
    """
    This function takes in the image as a numpy array and returns a numpy array of circles.
    :param img: Image as numpy array
    :return: Numpy array of circles.
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 5)
    img_circles = cv2.HoughCircles(img_blur,cv2.HOUGH_GRADIENT, 1,20,param1=100, param2=30,minRadius=1, maxRadius=30)
    return img_circles

def sign_axis(lines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    This function takes in a numpy array of lines and returns a tuple of np.ndarray and np.ndarray.

    This function should identify the lines that make up a sign and split the x and y coordinates.
    :param lines: Numpy array of lines.
    :return: Tuple of np.ndarray and np.ndarray with each np.ndarray consisting of the x coordinates and y coordinates
             respectively.
    """
    xaxis = np.empty(0, dtype=np.int32)
    yaxis = np.empty(0, dtype=np.int32)
    return xaxis, yaxis


def identify_traffic_light(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple identifying the location
    of the traffic light in the image and the lighted light.
    :param img: Image as numpy array
    :return: Tuple identifying the location of the traffic light in the image and light.
             ( x,   y, color)
             (140, 100, 'None') or (140, 100, 'Red')
             In the case of no light lit, coordinates can be just center of traffic light
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Corrected HSV ranges
    red_lower = np.array([0, 150, 100])
    red_upper = np.array([10, 255, 255])
    yellow_lower = np.array([20, 150, 100])
    yellow_upper = np.array([35, 255, 255])
    green_lower = np.array([40, 100, 100])
    green_upper = np.array([90, 255, 255])

    # Masks
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    red_circles = cv2.HoughCircles(red_mask, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=10, minRadius=5,maxRadius=40)
    yellow_circles = cv2.HoughCircles(yellow_mask, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=10, minRadius=5,maxRadius=40)
    green_circles = cv2.HoughCircles(green_mask, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=10, minRadius=5,maxRadius=40)

    def find_center(circles, color):
        if circles is not None:
            circles = np.uint16(np.around(circles))
            x, y, _ = circles[0][0]
            return (int(x), int(y), color)
        return None

    for mask, color in [(red_circles, 'Alex\'s Red Light'), (yellow_circles, 'Alex\'s Yellow Light'), (green_circles, 'Alex\'s Green Light')]:
        result = find_center(mask, color)
        if result is not None:
            return result

    # No light detected
    return (100, 100, 'None')


def identify_stop_sign(img: np.ndarray) -> tuple:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red_lower = np.array([0, 150, 70])
    red_upper = np.array([10, 255, 255])
    red_mask = cv2.inRange(hsv, red_lower, red_upper)

    # Apply Canny to the mask (numeric thresholds only)
    edges = cv2.Canny(red_mask, 50, 150)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            return (x, y, "stop")
    return (100, 100, "stop")


def identify_yield(img: np.ndarray) -> tuple:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red mask (wraps around hue=0)
    lower_red1 = np.array([0, 150, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 150, 100])
    upper_red2 = np.array([179, 255, 255])
    red_mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                              cv2.inRange(hsv, lower_red2, upper_red2))

    # Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # Approximate polygon
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        if len(approx) == 3:  # triangle
            # Optional: check orientation if needed
            M = cv2.moments(c)
            if M["m00"] != 0:
                x = int(M["m10"] / M["m00"])
                y = int(M["m01"] / M["m00"])
                return (x, y, "yield")

    return (100, 100, "yield")



def identify_construction(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'construction')
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    orange_mask = cv2.inRange(hsv, np.array([5, 150, 150]), np.array([20, 255, 255]))
    contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            return (x, y, "construction")
    return (100, 100, "construction")

def identify_warning(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'warning')
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, np.array([20, 150, 150]), np.array([35, 255, 255]))
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            return (x, y, "warning")
    return (100, 100, "warning")

def identify_rr_crossing(img: np.ndarray) -> tuple:
    """
    Detects the 'RR Crossing' sign using Hough Circle detection.
    Returns (x, y, 'rr_crossing') if found, else (100, 100, 'rr_crossing').
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)  # reduce noise

    # Hough Circle parameters might need tweaking
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=100
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, _ = circles[0][0]
        return (int(x), int(y), "rr_crossing")

    return (100, 100, "rr_crossing")



def identify_services(img: np.ndarray) -> tuple:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Looser blue range (good under noise or lighting changes)
    lower_blue = np.array([80, 255, 234])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Optional: small blur to merge nearby blue pixels
    mask = cv2.medianBlur(mask, 5)

    # Get all blue pixel coordinates
    ys, xs = np.where(mask > 0)

    if len(xs) > 0:
        x = int(xs.mean())
        y = int(ys.mean())
        return (x, y, "services")

    return (100, 100, "services")


def identify_signs(img: np.ndarray) -> np.ndarray:
    """
    This function takes in the image as a numpy array and returns a numpy array of all signs locations and name.
    Call the other identify functions to determine where that sign is if it exists.
    :param img: Image as numpy array
    :return: Numpy array of all signs locations and name.
             [[x, y, 'stop'],
              [x, y, 'construction']]
    """
    results = []
    for func in [identify_stop_sign, identify_construction, identify_yield,
                 identify_rr_crossing, identify_services, identify_warning,identify_traffic_light]:
        x, y, name = func(img)
        if name != "None":
            results.append((x, y, name))  # tuple per sign
    return tuple(results)  # return a tuple of tuples

def identify_signs_noisy(img: np.ndarray) -> tuple:
    """
    Handles noisy images with Gaussian noise using denoising, color normalization,
    and morphological cleaning, then calls the main identify_signs().
    """

    # --- Step 1: Denoise (handles Gaussian noise) ---
    img_denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # --- Step 2: Normalize brightness and contrast (LAB space) ---
    lab = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    img_normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # --- Step 3: Morphological cleaning ---
    kernel = np.ones((3, 3), np.uint8)
    img_clean = cv2.morphologyEx(img_normalized, cv2.MORPH_OPEN, kernel)
    img_clean = cv2.morphologyEx(img_clean, cv2.MORPH_CLOSE, kernel)

    # --- Step 4: Call your base sign detector ---
    return identify_signs(img_clean)


def identify_signs_real(img: np.ndarray) -> np.ndarray:
    """
    Handles real-world images using contrast enhancement, edge preservation, and adaptive preprocessing.
    """
    # Step 1: Preserve edges while smoothing noise
    img_filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 2: Adaptive contrast enhancement (CLAHE)
    lab = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Step 3: Mild sharpening to recover detail lost from filtering
    kernel_sharp = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])
    img_sharp = cv2.filter2D(img_clahe, -1, kernel_sharp)

    # Step 4: Slight blur to suppress residual pixel noise
    img_final = cv2.GaussianBlur(img_sharp, (3, 3), 0)

    # Step 5: Call your base sign detection
    return identify_signs(img_final)
