import cv2
import numpy as np

def nothing(x):
    pass

# Load your image
img = cv2.imread('images/tl_all_bright_default.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Create a window for trackbars
cv2.namedWindow("Trackbars")

# Create trackbars for hue, saturation, and value
cv2.createTrackbar("H Low", "Trackbars", 20, 179, nothing)
cv2.createTrackbar("H High", "Trackbars", 35, 179, nothing)
cv2.createTrackbar("S Low", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("S High", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("V Low", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("V High", "Trackbars", 255, 255, nothing)

while True:
    # Get current positions of all trackbars
    h_low = cv2.getTrackbarPos("H Low", "Trackbars")
    h_high = cv2.getTrackbarPos("H High", "Trackbars")
    s_low = cv2.getTrackbarPos("S Low", "Trackbars")
    s_high = cv2.getTrackbarPos("S High", "Trackbars")
    v_low = cv2.getTrackbarPos("V Low", "Trackbars")
    v_high = cv2.getTrackbarPos("V High", "Trackbars")

    # Create the mask using the selected range
    lower = np.array([h_low, s_low, v_low])
    upper = np.array([h_high, s_high, v_high])
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    # Show both the original and masked image
    cv2.imshow("Original", img)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
