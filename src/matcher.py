from matplotlib import pyplot as plt
import preprocess_all as pa
import eigenfeet
import numpy as np


#####################################################################################################################
# Matcher                                                                                                           #
# matching involves a simple distance metric in feet space with thresholding                                        #
#####################################################################################################################
def project_image(pca, flat_img):
    return pca.transform([flat_img])[0]

def manhattan_distances(test_vector, reference_vectors):
    return np.sum(np.abs(reference_vectors - test_vector), axis=1)


def match_image(test_vector, database_vectors, labels):
    distances = manhattan_distances(test_vector, database_vectors)
    min_index = np.argmin(distances)
    return labels[min_index], distances[min_index]

def preprocess_single_image(img_path, angle):
    img = pa.preprocess_image(img_path, angle)
    img = img / 255.0
    return img.flatten()

def verify_identity(test_vector, claimed_label, database_vectors, labels, threshold=100):
    matches = database_vectors[labels == claimed_label]
    if len(matches) == 0:
        return False, float('inf')
    
    distances = manhattan_distances(test_vector, matches)
    min_distance = np.min(distances)

    return min_distance < threshold, min_distance


if __name__ == '__main__':
    # Load and flatten new test image
    test_img_path = r'C:\Users\Andreas\Downloads\jku\bachelor\eigenfeet-biometrics\data\double\raw\user1\1-3.jpg'

    flat_test_img = preprocess_single_image(test_img_path)
    pca, X_pca, eigenfeet, X, y = eigenfeet.start(r'C:\Users\Andreas\Downloads\jku\bachelor\eigenfeet-biometrics\data\double\raw', r'C:\Users\Andreas\Downloads\jku\bachelor\eigenfeet-biometrics\data\double\preprocessed', r'C:\Users\Andreas\Downloads\jku\bachelor\eigenfeet-biometrics\data\double\pca_components')


    # Project it to PCA space
    test_pca_vector = project_image(pca, flat_test_img)

    # Match it to the database
    matched_label, distance = match_image(test_pca_vector, X_pca, y)

    print(f'Matched label: {matched_label}, Manhattan distance: {distance}')

    result, distance = verify_identity(test_pca_vector, "user9", X_pca, y, threshold=80)
    if result:
        print(f"Verification success! Distance: {distance}")
    else:
        print(f"Verification failed. Distance: {distance}")