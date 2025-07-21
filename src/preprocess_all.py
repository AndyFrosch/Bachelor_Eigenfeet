import os
import cv2
from matplotlib import pyplot as plt
import preprocessing as pp

#####################################################################################################################
# process 1 image                                                                                                   #
#####################################################################################################################
def preprocess_image(filepath, angle):
    # read image
    B = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if B is None:
        print(f"Warning: Could not read {filepath}")
        return None
    
    # canny edge detection
    B_1, threshold = pp.binarization(B)
    # fill interior
    B_2 = pp.fill_interior(B, B_1, threshold)
    # morphological_dilation
    B_4_eroded = pp.morphological_dilation(B_2)
    # compute_rotation_angle and crop image
    cropped_img = None
    if angle:
        rotated_bin, rotated_gray = pp.rotational_aligment(B, B_4_eroded)
        cropped_img = pp.bounding_box(rotated_gray, rotated_bin)
    else:
        cropped_img = pp.bounding_box(B, B_4_eroded)
    # resize image
    resized_img = pp.resize(cropped_img)
    return resized_img


#####################################################################################################################
# process all images and save them                                                                                  #
#####################################################################################################################
def preprocess(raw_dir, processed_dir, angle):
    os.makedirs(processed_dir, exist_ok=True)

    for user_folder in os.listdir(raw_dir):
        user_path = os.path.join(raw_dir, user_folder)
        print(user_path)
        if os.path.isdir(user_path):
            save_path = os.path.join(processed_dir, user_folder)
            os.makedirs(save_path, exist_ok=True)
            print(save_path)

            for image_file in os.listdir(user_path):
                if image_file.lower().endswith(('.jpg', '.jpeg')):
                    image_path = os.path.join(user_path, image_file)
                    processed_img = preprocess_image(image_path, angle)
                    
                    save_file = os.path.join(save_path, image_file)
                    cv2.imwrite(save_file, processed_img)
                break


# test
if __name__ == '__main__':
    pass
    # preprocess(raw_folder, preprocessed_folder)
