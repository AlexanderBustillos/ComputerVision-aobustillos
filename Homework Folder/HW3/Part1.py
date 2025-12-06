import cv2
import numpy as np

#Function to warp
def warp_image(input_path, output_path, src_pts, out_size):

    img = cv2.imread(input_path)
    img_copy = np.copy(img)
    w, h = out_size
    dst_pts = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst_pts)
    warped = cv2.warpPerspective(img_copy, M, (w, h))
    cv2.imwrite(output_path, warped)
    print(f"Saved: {output_path}")

quote_pts = np.array([[116, 68],[379, 152],[346, 429],[153, 376]])
warp_image("images/quote.jpg", "images/quote_warped.jpg", quote_pts, (400, 300))

quote2_pts = np.array([[116, 68],[411, 105],[471, 316],[193, 333]])
warp_image("images/quote2.jpg", "images/quote2_warped.jpg", quote2_pts, (400, 300))

sudoku_pts = np.array([[123, 235],[339, 170],[479, 328],[240, 435]])
warp_image("images/sudoku.jpg", "images/sudoku_warped.jpg", sudoku_pts, (500, 500))

program_pts = np.array([[46, 297],[292, 170],[545, 433],[306, 652]])
warp_image("images/program_sheet.jpg", "images/program_sheet_warped.jpg", program_pts, (500, 600))

image1 = cv2.imread("images/quote_warped.jpg")
cv2.imshow("Quote warped", image1)
image2 = cv2.imread("images/quote2_warped.jpg")
cv2.imshow("Quote2 warped", image2)
image3 = cv2.imread("images/sudoku_warped.jpg")
cv2.imshow("Sudoku warped", image3)
image4 = cv2.imread("images/program_sheet_warped.jpg")
cv2.imshow("Program sheet warped", image4)
while True:
    k = cv2.waitKey(1)
    if k == 27:
        break

cv2.destroyAllWindows()