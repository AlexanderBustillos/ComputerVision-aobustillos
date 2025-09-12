import random

import cv2
import numpy as np


def read_image(image_path: str) -> np.ndarray:

    """
     This function reads an image and returns it as a numpy array
     :param image_path: String of path to file
     :return img: Image array as ndarray
     """
    img = cv2.imread(image_path) #basic img reading, taking in the path and reading it
    return np.copy(img)


def extract_green(img: np.ndarray) -> np.ndarray:
    """
        This function takes an image and returns the green channel
        :param img: Image array as ndarray
        :return: Image array as ndarray of just green channel
        """
    g = img.copy() #copying the given image as to not affect the original
    g= g[:,:,1] #this takes all the green from the rows and columns of pixels and should return the green channel
    #channels are RGB conventially but CVS uses BGR instead so B is channel 0, green stays channel 1 and red is channel 2
    return np.copy(g)


def extract_red(img: np.ndarray) -> np.ndarray:
    """
       This function takes an image and returns the red channel
       :param img: Image array as ndarray
       :return: Image array as ndarray of just red channel
       """
    r = img.copy()
    r= r[:,:,2] #this takes all the red from the rows and columns of pixels and should return the red channel
    return np.copy(r)


def extract_blue(img: np.ndarray) -> np.ndarray:
    """
       This function takes an image and returns the blue channel
       :param img: Image array as ndarray
       :return: Image array as ndarray of just blue channel
       """
    b = img.copy()
    b = b[:,:,0] #this takes all the blue from the rows and columns of pixels and should return the blue channel
    return np.copy(b)


def swap_red_green_channel(img: np.ndarray) -> np.ndarray:
    """
       This function takes an image and returns the image with the red and green channel
       :param img: Image array as ndarray
       :return: Image array as ndarray of red and green channels swapped
       """
    imgswap = img.copy() #coppying the given img
    red = imgswap[:,:,2].copy() # copying all the values in the red channel
    green = imgswap[:,:,1].copy()#copying all the value in the green channel
    imgswap[:,:,2] = green #setting all the red to green
    imgswap[:,:,1] = red    #setting all the green to red
    return np.copy(imgswap)


def embed_middle(image1: np.ndarray, image2: np.ndarray, embed_size: (int, int)) -> np.ndarray:
    """
      This function takes two images and embeds the embed_size pixels from img2 onto img1
      :param img1: Image array as ndarray
      :param img2: Image array as ndarray
      :param embed_size: Tuple of size (width, height)
      :return: Image array as ndarray of img1 with img2 embedded in the middle
      """

    img1 = image1.copy() #copying image 1
    img2 = image2.copy() #copying image 2
    center_y = img1.shape[0] // 2 # getting the shape so we can get the middle row of the image height
    center_x = img1.shape[1] // 2   # getting the shape so we can get the middle row of the image width

    x = embed_size[0] // 2 # this is because we are taking in 60 pixles but technically we want 30 pixles up and 30 down to get a 60x60
    y = embed_size[1] // 2
    img_slice = img1[center_y-x:center_y+y, center_x-x :center_x+y,:] #extracting the slice of the image should be a 60x60 pixels

    center2_y = img2.shape[0] // 2 # taking the shape of the second img
    center2_x = img2.shape[1] // 2

    img2[center2_y-x:center2_y+y, center2_x-x :center2_x+y,:] = img_slice # placing the image slice on top of the middle of the second image

    return np.copy(img2)


def calc_stats(img: np.ndarray) -> np.ndarray:
    """
    This function takes an image and returns the mean and standard deviation
    :param img: Image array as ndarray
    :return: Numpy array with mean and standard deviation in that order
    """

    imgcalc = img.copy() #copying given img
    mean, std = cv2.meanStdDev(imgcalc) # getting the mean and standard deviation of the image
    return np.array([float(mean[0][0]), float(std[0][0])], dtype=np.float64) # returning an array of the mean and standard deviation, 64 bits allows for room


def shift_image(img: np.ndarray, shift_val: int) -> np.ndarray:
    """
    This function takes an image and returns the image shifted by shift_val pixels to the right.
    Should have an appropriate border for the shifted area:
    https://docs.opencv.org/3.4/dc/da3/tutorial_copyMakeBorder.html

    Returned image should be the same size as the input image.
    :param img: Image array as ndarray
    :param shift_val: Value to shift the image
    :return: Shifted image as ndarray
    """
    x = int(shift_val) #getting the int of the shift val, should already be int but this works so.
    h, w = img.shape # getting the height and width of the img
    imgshift = img.copy() # copying the image
    imgshift = cv2.copyMakeBorder(imgshift,0,0,x,0,cv2.BORDER_REPLICATE,value=0) # making sure we replicate the border that is going to be shifted to the right, i chose replicate because its better then a black side
    imgshift = imgshift[:,:w] #cropping
    return np.copy(imgshift)


def difference_image(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    This function takes two images and returns the first subtracted from the second

    Make sure the image to return is normalized:
    https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga87eef7ee3970f86906d69a92cbf064bd

    :param img1: Image array as ndarray
    :param img2: Image array as ndarray
    :return: Image array as ndarray
    """
    image1 = img1.copy()
    image2 = img2.copy()
    imgdiff = image1 - image2 #subtrating the original with the shifted, both are gray scale
    normimg = cv2.normalize(imgdiff,None,0,255,cv2.NORM_MINMAX) #normalizing the img difference
    return np.copy(normimg)


def add_channel_noise(img: np.ndarray, channel: int, sigma: int) -> np.ndarray:
    """
    This function takes an image and adds noise to the specified channel.

    Should probably look at randn from numpy

    Make sure the image to return is normalized:
    https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga87eef7ee3970f86906d69a92cbf064bd

    :param img: Image array as ndarray
    :param channel: Channel to add noise to
    :param sigma: Gaussian noise standard deviation
    :return: Image array with gaussian noise added
    """
    noisyimg = img.copy().astype(np.float32) #copying the image as a float
    gaussian = np.random.normal(0, sigma, (img.shape[0], img.shape[1])) #adding random noise with a set sigma
    noisyimg[:, :, channel] += gaussian #adding the noise to one channel specifically
    norming = cv2.normalize(noisyimg, None, 0, 255, cv2.NORM_MINMAX) #normalizing
    return np.copy(norming.astype(np.uint8))

def add_salt_pepper(img: np.ndarray) -> np.ndarray:
    """
    This function takes an image and adds salt and pepper noise.

    Must only work with grayscale images
    :param img: Image array as ndarray
    :return: Image array with salt and pepper noise
    """
    saltpepperimg = img.copy()
    h, w = saltpepperimg.shape #getting the height and width of the img
    numofpixels = 5000  #num of black and white pixels
    #this gets a random coordinate and makes it white or black
    for i in range(numofpixels):

        y_coord = random.randint(0, h - 1)
        x_coord = random.randint(0, w - 1)

        saltpepperimg[y_coord][x_coord] = 255

    for i in range(numofpixels):

        y_coord = random.randint(0, h - 1)
        x_coord = random.randint(0, w - 1)

        saltpepperimg[y_coord][x_coord] = 0

    return np.copy(saltpepperimg)


def blur_image(img: np.ndarray, ksize: int) -> np.ndarray:
    """
    This function takes an image and returns the blurred image

    https://docs.opencv.org/4.x/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html
    :param img: Image array as ndarray
    :param ksize: Kernel Size for medianBlur
    :return: Image array with blurred image
    """
    medBlurimg = img.copy()
    #standered blur with k size very simple
    medBlurimg = cv2.medianBlur(medBlurimg, ksize, 0)

    return np.copy(medBlurimg)
