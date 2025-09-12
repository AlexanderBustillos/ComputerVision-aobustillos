import os
import cv2
from hw1 import *


def main() -> None:
    # TODO: Add in images to read
    img1 = read_image("images/hw1_pic1.jpg")  # calling function read_image should make a copy of the original image just in case
    img2 = read_image("images/hw1_pic2.jpg")

    # TODO: replace None with the correct code to convert img1 and img2
    hw1_pic1_gray = img1.copy()  # making a copy of a copy just for added layer of convolution, confusion and concern
    hw1_pic1_gray = cv2.cvtColor(hw1_pic1_gray, cv2.COLOR_BGR2GRAY)  # changing the image to gray scale
    hw1_pic2_gray = img2.copy()
    hw1_pic2_gray = cv2.cvtColor(hw1_pic2_gray, cv2.COLOR_BGR2GRAY)

    hw1_pic1_hsv = img1.copy()
    hw1_pic1_hsv = cv2.cvtColor(hw1_pic1_hsv, cv2.COLOR_BGR2HSV)
    hw1_pic2_hsv = img2.copy()
    hw1_pic2_hsv = cv2.cvtColor(hw1_pic2_hsv, cv2.COLOR_BGR2HSV)

    hw1_pic1_red = extract_red(img1)
    hw1_pic1_green = extract_green(img1)
    hw1_pic1_blue = extract_blue(img1)

    hw1_pic2_red = extract_red(img2)
    hw1_pic2_green = extract_green(img2)
    hw1_pic2_blue = extract_blue(img2)

    hw1_pic1_swapped = swap_red_green_channel(img1)
    hw1_pic2_swapped = swap_red_green_channel(img2)

    hw1_embedded = embed_middle(img1, img2, embed_size=(60, 60))

    hw1_pic1_stats = calc_stats(hw1_pic1_gray)
    hw1_pic2_stats = calc_stats(hw1_pic2_gray)

    # TODO: Replace None with correct calls
    hw1_pic1_shift =shift_image(hw1_pic1_gray, 2)
    hw1_pic2_shift =shift_image(hw1_pic2_gray, 2)

    # TODO: Replace None with correct calls. The difference should be between the original and shifted image
    hw1_pic1_diff = difference_image(hw1_pic1_gray, hw1_pic1_shift)
    hw1_pic2_diff = difference_image(hw1_pic2_gray, hw1_pic2_shift)

    # TODO: Select appropriate sigma and call functions
    sigma = 50

    hw1_pic1_noise_red = add_channel_noise(img1 ,2, sigma)
    hw1_pic1_noise_green = add_channel_noise(img1 ,1, sigma)
    hw1_pic1_noise_blue = add_channel_noise(img1 ,0, sigma)

    hw1_pic2_noise_red = add_channel_noise(img2, 2, sigma)
    hw1_pic2_noise_green = add_channel_noise(img2, 1, sigma)
    hw1_pic2_noise_blue = add_channel_noise(img2, 0, sigma)

    hw1_pic1_spnoise = add_salt_pepper(hw1_pic1_gray)
    hw1_pic2_spnoise = add_salt_pepper(hw1_pic2_gray)

    # TODO: Select appropriate ksize, must be odd
    ksize = 11
    hw1_pic1_blur = blur_image(hw1_pic1_spnoise, ksize)
    hw1_pic2_blur = blur_image(hw1_pic2_spnoise, ksize)

    # TODO: Write out all images to appropriate files
    # load images

    cv2.imshow("Original image 1", img1)
    cv2.imshow("Original image 2", img2)
    # gray scale of each img
    cv2.imshow("Gray Scale image 1", hw1_pic1_gray)  # display image
    cv2.imshow("Gray Scale image 2", hw1_pic2_gray)

    # hsv of each img
    cv2.imshow("HSV image 1", hw1_pic1_hsv)
    cv2.imshow("HSV image 2", hw1_pic2_hsv)

    # extracting red
    cv2.imshow("Red image 1", hw1_pic1_red)  # display image
    cv2.imshow("Red image 2", hw1_pic2_red)  # display image

    # extracting green
    cv2.imshow("Green image 1", hw1_pic1_green)  # display image
    cv2.imshow("Green image 2", hw1_pic2_green)  # display image

    # extracting blue
    cv2.imshow("Blue image 1", hw1_pic1_blue)  # display image
    cv2.imshow("Blue image 2", hw1_pic2_blue)  # display image

    # swapping red and green channels
    cv2.imshow("Swapping red and green channels image 1", hw1_pic1_swapped)  # display image
    cv2.imshow("Swapping red and green channels image 2", hw1_pic2_swapped)  # display image
    # taking the pixles and pasting them elsewhere
    cv2.imshow("Image 1 middle 60x60 pixles sliced and moved to the middle of image2", hw1_embedded)

    #Shows mean and standard deviation of img
    print("Mean and standard deviation of pic1: ",hw1_pic1_stats)
    print("Mean and standard deviation of pic2: ",hw1_pic2_stats)

    #Shifted image to the right by 2 pixels and then replicating border
    cv2.imshow("Shifted right image 1", hw1_pic1_shift)
    cv2.imshow("Shifted right image 2", hw1_pic2_shift)

    #The difference of the images then normalized
    cv2.imshow("Differance of img 1", hw1_pic1_diff)
    cv2.imshow("Differance of img 2", hw1_pic2_diff)

    #adding noise to different channels
    cv2.imshow("Adding noise to red image 1", hw1_pic1_noise_red)
    cv2.imshow("Adding noise to green image 1", hw1_pic1_noise_green)
    cv2.imshow("Adding noise to blue image 1", hw1_pic1_noise_blue)

    cv2.imshow("Adding noise to red image 2", hw1_pic2_noise_red)
    cv2.imshow("Adding noise to green image 2", hw1_pic2_noise_green)
    cv2.imshow("Adding noise to blue image 2", hw1_pic2_noise_blue)

    #salt and pepper filter on the grayscale

    cv2.imshow("Salt and pepper of image 1", hw1_pic1_spnoise)
    cv2.imshow("Salt and pepper of image 2", hw1_pic2_spnoise)

    #Applying the median blur to the salt and pepper image
    cv2.imshow("Median blur to sp image 1", hw1_pic1_blur)
    cv2.imshow("Median blur to sp image 2", hw1_pic2_blur)


    cv2.waitKey(0)

if __name__ == '__main__':
    main()