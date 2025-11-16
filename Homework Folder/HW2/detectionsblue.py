import cv2
import numpy as np

def nothing(x):
    pass

# Load image
img = cv2.imread('images/all_signs_noise.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Create window and trackbars for adjusting blue HSV range
cv2.namedWindow("Blue Trackbars")
cv2.resizeWindow("Blue Trackbars", 600, 300)

cv2.createTrackbar("H Low", "Blue Trackbars", 90, 179, nothing)
cv2.createTrackbar("H High", "Blue Trackbars", 130, 179, nothing)
cv2.createTrackbar("S Low", "Blue Trackbars", 50, 255, nothing)
cv2.createTrackbar("S High", "Blue Trackbars", 255, 255, nothing)
cv2.createTrackbar("V Low", "Blue Trackbars", 50, 255, nothing)
cv2.createTrackbar("V High", "Blue Trackbars", 255, 255, nothing)

while True:
    # Get trackbar positions
    h_low = cv2.getTrackbarPos("H Low", "Blue Trackbars")
    h_high = cv2.getTrackbarPos("H High", "Blue Trackbars")
    s_low = cv2.getTrackbarPos("S Low", "Blue Trackbars")
    s_high = cv2.getTrackbarPos("S High", "Blue Trackbars")
    v_low = cv2.getTrackbarPos("V Low", "Blue Trackbars")
    v_high = cv2.getTrackbarPos("V High", "Blue Trackbars")

    # Create mask for blue pixels
    lower = np.array([h_low, s_low, v_low])
    upper = np.array([h_high, s_high, v_high])
    mask = cv2.inRange(hsv, lower, upper)

    # Apply mask to original image
    result = cv2.bitwise_and(img, img, mask=mask)

    # Display images
    cv2.imshow("Original", img)
    cv2.imshow("Mask", mask)
    cv2.imshow("Blue Pixels", result)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
