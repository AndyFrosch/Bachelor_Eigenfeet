import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt

test = False
#####################################################################################################################
#-------------------------------------------------- agenda ---------------------------------------------------------#
#####################################################################################################################
# take raw images and preprocess them so they can be used for PCA to compute eigenfeet                              #
# ->                                                                                                                #         
# ->                                                                                                                #
# ->                                                                                                                #
# how do we do this:                                                                                                #
# all of this is based on 'Personal identification using Eigenfeet, Ballprint and Foot geometry biometrics'         #
# published by Andreas Uhl and Peter Wild                                                                           #
# Author Andreas Froschauer k12103798                                                                               #
#####################################################################################################################
#------------------------------------------------ Procedures -------------------------------------------------------#
#####################################################################################################################
# 1. seperate foot and background                                                                                   #
# binarization using Canny edge detection                                                                           #
# then apply binary thresholding on original image B -> we get B_1                                                  #
# problem: background isn't ideal, might have to manually edit images, or find a way to automate?                   #
#####################################################################################################################
def binarization(B):
    assert B is not None, "file could not be read, check with os.path.exists()"
    # i decided to add a simple gaussian filter before starting with edge detection                                
    # it improves results significantly since our sample data has a lot of noise in it                             
    B = cv2.GaussianBlur(B, (5, 5), 1.5)
    b = 77
    B_1 = cv2.Canny(B, b, 200)
    
    # test
    if test:
        show(B_1, "image after binarization using Canny edge detection")

    return B_1, b
#####################################################################################################################
# 2. now we have B_1                                                                                                #
# fill interior using binary thresholding on B i.e.                                                                 #
# B_2(x,y) = max(bin_b(B)(x,y), B_1(x,y)) --- bin_b(B) is binarization of B using threshold b                       #
#####################################################################################################################
def fill_interior(B, B_1, b):
    assert B_1 is not None, "no edge image"
    _, B_2 = cv2.threshold(B, b, 255, cv2.THRESH_BINARY)
    B_2 = cv2.bitwise_or(B_2, B_1)
    # test
    if test:
        show(B_2, "image after filling interior using binary thresholding")

    return B_2
#####################################################################################################################
# 3. now we have B_2                                                                                                #
# now apply morphological dilation on B_2                                                                           #
# B_3 = B_2 xor S --- S= {(x,y)| S_xy intersection  B_2 != {}} --- s_xy is the shift of S by (x,y)                  #
# then remove small BLOBS (noise outside the foot), fill black BLOBS white (inside the foot)                        #
# this yields B_4                                                                                                   #
# to get B_5 apply morphological erosion                                                                            #
# B_5 = B_4 xor S --- S = {(x,y)| S_xy is a subset of B_4}                                                          #
#####################################################################################################################
def remove_small_white_blobs(img, min_area=500):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img)
    cleaned = np.zeros_like(img)

    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255

    return cleaned

def fill_black_blobs(img):
    inv_img = cv2.bitwise_not(img)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv_img)
    areas = stats[1:, cv2.CC_STAT_AREA]

    if len(areas) == 0:
        return img
    largest_label = 1 + np.argmax(areas)
    filled = img.copy()

    for label in range(1, num_labels):
        if label != largest_label:
            filled[labels == label] = 255

    return filled
    

def morphological_dilation(B_2):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) # size changes how much dilation is happening
    B_3 = cv2.dilate(B_2, kernel)

    # Remove small white blobs after dilation
    B_3_clean = remove_small_white_blobs(B_3, min_area=10000)
    
    # Fill black blobs
    B_4 = fill_black_blobs(B_3_clean)

    # Erosion
    B_4_eroded = cv2.erode(B_4, kernel)
    # _ = cv2.bitwise_xor(B_4, B_4_eroded) # we dont need this for pca

    # test
    if test:
        show(B_4_eroded, "image after erosion")

    return B_4_eroded
#####################################################################################################################
# 4. rotational alignment                                                                                           #
# use B_4 to find the best-fitting ellipse,                                                                         #
# then find the angle theta between y-axis and major axis of the best matching ellipse                              #
# calculate center of mass C=(xbar,ybar)                                                                            #
#####################################################################################################################
def compute_rotation_angle(binary_img):
    B = (binary_img > 0).astype(np.float32)
    n, m = B.shape
    i, j = np.meshgrid(np.arange(n), np.arange(m), indexing='ij')

    A = np.sum(B)
    assert not A == 0, "Empty binary image; no white pixels found."

    x_bar = np.sum(j * B) / A
    y_bar = np.sum(i * B) / A

    # Shift coordinates by centroid
    x_ = j - x_bar
    y_ = i - y_bar

    mu_20 = np.sum((x_ ** 2) * B)
    mu_11 = np.sum((x_ * y_) * B)
    mu_02 = np.sum((y_ ** 2) * B)

    theta = 0.5 * np.arctan2(2 * mu_11, mu_20 - mu_02)

    angle_deg = np.degrees(theta)

    # Normalize angle to [0, 180)
    angle_deg = angle_deg % 180

    return angle_deg, (x_bar, y_bar)

def rotational_aligment(B, B_4_eroded):
    angle, center = compute_rotation_angle(B_4_eroded)
    # Rotation center needs to be integer and swapped because image coordinates are (y,x)
    center_int = (int(center[0]), int(center[1]))

    M = cv2.getRotationMatrix2D(center_int, angle - 90, 1.0)
    rotated_binary = cv2.warpAffine(B_4_eroded, M, (B_4_eroded.shape[1], B_4_eroded.shape[0]), flags=cv2.INTER_LINEAR)
    rotated_gray = cv2.warpAffine(B, M, (B.shape[1], B.shape[0]), flags=cv2.INTER_LINEAR)

    # test
    if test:
        show(rotated_gray, "image after rotation")

    return rotated_binary, rotated_gray
#####################################################################################################################
# 5. masking + bounding box                                                                                         #
# for PCA we need to get the region of interest i.e. the footprint without any background                           #
# therefore we apply our mask onto the grayscaled original image                                                    #
#####################################################################################################################
def mask_image(rotated_gray, rotated_binary):
    mask_binary = (rotated_binary > 0).astype(np.uint8)
    foot_only_img = rotated_gray * mask_binary

    return mask_binary, foot_only_img

def bounding_box(rotated_gray, rotated_binary):
    mask_binary, foot_only_img = mask_image(rotated_gray, rotated_binary)
    coords = np.column_stack(np.where(mask_binary == 1))
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    cropped_img = foot_only_img[y_min:y_max+1, x_min:x_max+1]

    # test
    if test:
        show(cropped_img, "image after bounding box")
    
    return cropped_img
#####################################################################################################################
# 6. resize image to 128x256                                                                                        #
#####################################################################################################################
def resize(img, size=(128, 256)):
    """
    Resize the input image to the given size (width, height).
    OpenCV expects size as (width, height).
    """
    resized_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    
    # test
    if test:   
        show(resized_img, "image after resize")
    
    return resized_img
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
def show(img, title):
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(title, fontsize=14)
    plt.show()
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#



# quick functionality test 
if __name__ == '__main__':
    # pass
    test = True


    if test:
        B = cv2.imread(r'C:\Users\Andreas\Downloads\jku\bachelor\eigenfeet-biometrics\data\single\raw\user1\1-1.jpg', cv2.IMREAD_GRAYSCALE)

        B_1, threshold = binarization(B)
        B_2 = fill_interior(B, B_1, threshold)
        B_4_eroded = morphological_dilation(B_2)
        rotated_bin, rotated_gray = rotational_aligment(B, B_4_eroded)
        bb = bounding_box(rotated_gray, rotated_bin)


        final = resize(bb)

        
        show(final, "processed image")