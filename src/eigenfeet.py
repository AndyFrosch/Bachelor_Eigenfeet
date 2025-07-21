import math
import os
from typing import Tuple
import numpy as np
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import preprocess_all as pa
import preprocessing as pp


# Set your image folder
# preprocessed_folder   = r'C:\Users\Andreas\Downloads\jku\bachelor\eigenfeet-biometrics\data\preprocessed'
image_shape = (128, 256)
# n_components = 20
#####################################################################################################################
# Implementation of PCA Algorithm
# Based on the structure provided in 'Personal identification using Eigenfeet,                                      #
# Ballprint and Foot geometry biometrics'                                                                           #
# published by Andreas Uhl and Peter Wild                                                                           #
# Author Andreas Froschauer k12103798                                                                               #
#####################################################################################################################
# 1. Acquisition -> steps:                                                                                          #
# - center training data                                                                                            #
# - subtract average vector                                                                                         #
#####################################################################################################################
# 2. Computation -> steps:                                                                                          #
# - compute covariance matrix mn x mn                                                                               #
# - compute eigenvectors                                                                                            #
# - compute eigenvalues                                                                                             #
#####################################################################################################################
# 3. Ordering and Selection                                                                                         #
# order the resulting eigenvectors and take L eigenvectors + eigenvalues that explain the variance best             #
#####################################################################################################################
# 4. Feature Extraction -> steps:                                                                                   #
# - normalization of foot vector                                                                                    #
# - project foot vector onto eigenspace to get feature vector which consist of L components                         #
# 5. Matcher                                                                                                        #
# match images with the manhatten distance                                                                          #
#####################################################################################################################


def fetch_images(preprocessed):
    images = []
    labels = []
    for user_id in os.listdir(preprocessed):
        path = os.path.join(preprocessed, user_id)
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.lower().endswith(('.jpg', '.jpeg')):
                    img_path = os.path.join(path, file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    # pp.show(img, 's')
                    if img is None:
                        continue
                    img = img / 255.0
                    
                    images.append(img.flatten())
                    labels.append(user_id)
                    shape = img.shape
                    
    return np.array(images), np.array(labels), shape


# Apply PCA to the dataset
def pca_model(X, n_components, image_shape) -> Tuple[PCA, np.ndarray, np.ndarray]:
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=564263)
    X_pca = pca.fit_transform(X)
    print(image_shape)
    eigenfeet = pca.components_.reshape((n_components, *image_shape))

    return pca, X_pca, eigenfeet


def prepare_data(raw, preprocessed, angle):
    print('Starting preprocessing:')
    pa.preprocess(raw, preprocessed, angle)
    print('Preprocessing done')

    

def save_pca(eigenfeet, pca_dir):
    os.makedirs(pca_dir, exist_ok=True)
    n = len(eigenfeet)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
    for i, ax in enumerate(axes.flat):
        if i == n: break
        ax.imshow(eigenfeet[i], cmap='gray')
        ax.axis('off')
        ax.text(0.5, -0.1, str(f'component {i+1}'), transform=ax.transAxes,
            ha='center', va='top', fontsize=10)
    plt.suptitle("Eigenfeet", fontsize=16)

    # Save to file instead of showing
    save_path = os.path.join(pca_dir, 'top_eigenfeet.png')
    plt.savefig(save_path)
    plt.close(fig)  # Close the figure to free memory

    os.makedirs(pca_dir, exist_ok=True)
    for i, component in enumerate(eigenfeet):
        # Normalize the data to 0–255 for image saving (if needed)
        norm_component = cv2.normalize(component, None, 0, 255, cv2.NORM_MINMAX)
        norm_component = norm_component.astype(np.uint8)

        filename = f'component_{i}.png'
        file_path = os.path.join(pca_dir, filename)

        cv2.imwrite(file_path, norm_component)



def start(raw, preprocessed, pca_dir, n_components=20, skip_preprocessing=True, 
          double_single=True, angle=True) -> Tuple[PCA, np.ndarray, np.ndarray]:
    if not skip_preprocessing:
        prepare_data(raw, preprocessed, angle)
        print('preparation done')

    X, y, image_shape = fetch_images(preprocessed)
    pca, X_pca, eigenfeet = pca_model(X, n_components, image_shape)
    save_pca(eigenfeet, pca_dir)
    return pca, X_pca, eigenfeet, X, y



if __name__ == '__main__':
    pass