import cv2
import numpy as np


def main() -> None:
    #Opening the image with imread
    img = cv2.imread("images\OpenCV_Logo.jpg", 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("image", img)
    cv2.waitKey(0)

if __name__ == '__main__':
   main()