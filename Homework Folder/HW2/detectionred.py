import cv2
import numpy as np

def nothing(x):
    pass

# --- Load an image ---
# Replace this path with your test image (a traffic light picture)
img = cv2.imread('images/tl_all_bright_default.jpg')
if img is None:
    raise FileNotFoundError("Image not found. Please check your path.")

# --- Convert once to HSV ---
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# --- Create window ---
cv2.namedWindow('Trackbars')

# --- Create trackbars for HSV lower and upper limits ---
cv2.createTrackbar('LH', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('LS', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('LV', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('UH', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('US', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('UV', 'Trackbars', 255, 255, nothing)

while True:
    # --- Get trackbar positions ---
    lh = cv2.getTrackbarPos('LH', 'Trackbars')
    ls = cv2.getTrackbarPos('LS', 'Trackbars')
    lv = cv2.getTrackbarPos('LV', 'Trackbars')
    uh = cv2.getTrackbarPos('UH', 'Trackbars')
    us = cv2.getTrackbarPos('US', 'Trackbars')
    uv = cv2.getTrackbarPos('UV', 'Trackbars')

    # --- Define lower/upper HSV ---
    lower = np.array([lh, ls, lv])
    upper = np.array([uh, us, uv])

    # --- Mask based on range ---
    mask = cv2.inRange(hsv, lower, upper)

    # --- Apply mask ---
    result = cv2.bitwise_and(img, img, mask=mask)

    # --- Show the results ---
    cv2.imshow('Original', img)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)

    # --- Exit on ESC key ---
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()
