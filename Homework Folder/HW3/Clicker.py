
import cv2

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)

img = cv2.imread("images/quote.jpg")
cv2.imshow("Click to print coordinates", img)
cv2.setMouseCallback("Click to print coordinates", click)

cv2.waitKey(0)
cv2.destroyAllWindows()
