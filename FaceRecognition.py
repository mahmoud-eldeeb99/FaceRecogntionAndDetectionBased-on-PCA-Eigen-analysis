import os
import glob
from sklearn import preprocessing
import cv2
import numpy as np
import math


def Dataset_Info_Extractor(dataset_path, num_of_imgs=0, shape=(112, 92)):
    names = list()
    i = 0
    # Calculate the number of the image in the data set
    for images in glob.glob(dataset_path + '/**', recursive=True):  # Loop through all the images in the folder
        if images[-3:] == 'pgm' or images[-3:] == 'jpg':
            num_of_imgs += 1

    print('num of imges ',num_of_imgs)
    all_img = np.zeros((num_of_imgs, shape[0], shape[1]), dtype='float64')
    for folder in glob.glob(dataset_path + '/*'):  # Loop through folders
        for _ in range(10):
            names.append(folder[-3:].replace('/', ''))
        # read the image
        for image in glob.glob(folder + '/*'):
            read_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(read_image, (shape[1], shape[0]))
            all_img[i] = np.array(resized_image)
            i += 1
    # print(names)
    # print(all_img)
    return num_of_imgs, names, all_img

def Processing(num_of_imgs, all_img,shape=(112, 92)):
    # Calculating the mean
    original_img = np.resize(all_img, (num_of_imgs, shape[0]*shape[1])) ## stck the imges into  2d arry
    mean_vector = np.sum(original_img, axis=0, dtype='float64')/num_of_imgs
    mean_matrix = np.tile(mean_vector, (num_of_imgs, 1))
    # zero mean img
    zero_mean_img = original_img - mean_matrix
    # Calculate eigen values and eigen vector
    Sym_matrix = (zero_mean_img.dot(zero_mean_img.T))/num_of_imgs ## coneriancec matrix
    # Calculating the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(Sym_matrix)
    #sort eigenvalues in descending order z
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    #Sort the eigenvectors according to the highest eigenvalues
    eigenvectors = eigenvectors[:,idx]
    # perform linear combination with Matrix A_tilde
    eigenvector_C = zero_mean_img.T @ eigenvectors
    print(eigenvector_C.shape)#Each column is an eigenvector
    #Normalize the vector
    eigenfaces = preprocessing.normalize(eigenvector_C.T)
    print(eigenfaces.shape)

    return eigenfaces, mean_vector, zero_mean_img

def Face_recognetion(test_img,eigenfaces, mean_vector,num_of_imgs, names, zero_mean_img, shape=(112, 92)):

    test_img = cv2.resize(test_img,(shape[1],shape[0]))
    mean_sub_testimg = np.reshape(test_img, (test_img.shape[0]*test_img.shape[1])) - mean_vector

    q = 350   # 350 eigenvectors is chosen
    E = eigenfaces[:q].dot(mean_sub_testimg)
    reconstruction = eigenfaces[:q].T.dot(E)

    """""
    # Detect Face
    thres_1 = 3000 # Chosen threshold to detect face
    # Perform Linear combination for the new face space
    projected_new_img_vect=eigenfaces[:q].T @ E
    diff = mean_sub_testimg-projected_new_img_vect
    # Find the difference between the projected test image vector and the mean vector of the images
    beta = math.sqrt(diff.dot(diff))
    if beta<thres_1:
        print("Face Detected in the image!", beta)
    else:
        print("No face Detected in the image!", beta)
     """""

    # Classify the image belongs to which class (recognition)
    thres_2 = 3000
    smallest_value = None
    index = None
    Face_found = None
    for z in range(num_of_imgs):
        E_z = eigenfaces[:q].dot(zero_mean_img[z])
        diff = E-E_z
        epsilon_z = math.sqrt(diff.dot(diff))
        if smallest_value==None:
            smallest_value=epsilon_z
            index = z
        if smallest_value>epsilon_z:
            smallest_value=epsilon_z
            index = z
    if smallest_value<thres_2:
        print(smallest_value, names[index])
        Face_found = True
    else:
        print(smallest_value,"unknown Face")
        Face_found = False

    return Face_found
# test
# dataset_path = os.getcwd() + '/FaceDataset/'
# test_img = cv2.imread('images/1.pgm', cv2.IMREAD_GRAYSCALE)
# num_of_imgs, names, all_img = Dataset_Info_Extractor(dataset_path)
# eigenfaces, mean_vector, zero_mean_img = Processing(num_of_imgs, all_img)
# Face_recognetion(test_img,eigenfaces, mean_vector, num_of_imgs, names, zero_mean_img)