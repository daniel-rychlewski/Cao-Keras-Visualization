import os
import sys

import numpy as np
import scipy.io as sio
from keras.utils import np_utils
from sklearn import preprocessing

import compression
import preprocessing as pp
from generate_model import pretty_print_count


def load_data():
    data_path = os.path.join(os.getcwd(),'.')
    data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
    train_labels = np.load("train_data.npy")
    test_labels = np.load("test_data.npy")
    
    return data, train_labels, test_labels

def patch_1dim_split(X, train_data, test_data, PATCH_SIZE):
    padding = int((PATCH_SIZE - 1) / 2) #Patch de 3*3 = padding de 1 (centre + 1 de chaque coté)
    #X_padding = np.zeros(X)
    X_padding = np.pad(X, [(padding, padding), (padding, padding), (0, 0)], mode='constant')
    
    X_patch = np.zeros((X.shape[0] * X.shape[1], PATCH_SIZE, PATCH_SIZE, X.shape[2]))
    y_train_patch = np.zeros((train_data.shape[0] * train_data.shape[1]))
    y_test_patch = np.zeros((test_data.shape[0] * test_data.shape[1]))
    
    index = 0
    for i in range(0, X_padding.shape[0] - 2 * padding):
        for j in range(0, X_padding.shape[1] - 2 * padding):
            # This condition is for less frequent updates. 
            if i % 8 == 0 or index == (X_padding.shape[0] - 2 * padding) * (X_padding.shape[1] - 2 * padding) - 1:
                print_progress_bar(index + 1, (X_padding.shape[0] - 2 * padding) * (X_padding.shape[1] - 2 * padding))
            patch = X_padding[i:i + 2 * padding + 1, j:j + 2 * padding + 1]
            X_patch[index, :, :, :] = patch
            y_train_patch[index] = train_data[i, j]
            y_test_patch[index] = test_data[i, j]
            index += 1
    
    print("\nCreating train/test arrays and removing zero labels...")
    print_progress_bar(1, 7)
    X_train_patch = np.copy(X_patch)
    print_progress_bar(2, 7)
    X_test_patch = np.copy(X_patch)
    
    print_progress_bar(3, 7)
    X_train_patch = X_train_patch[y_train_patch > 0,:,:,:]
    print_progress_bar(4, 7)
    X_test_patch = X_test_patch[y_test_patch > 0,:,:,:]
    print_progress_bar(5, 7)
    y_train_patch = y_train_patch[y_train_patch > 0] - 1
    print_progress_bar(6, 70)
    y_test_patch = y_test_patch[y_test_patch > 0] - 1
    print_progress_bar(7, 7)
    print("Done.")
    
    return X_train_patch, X_test_patch, y_train_patch, y_test_patch

def dimensionality_reduction(X, compression, numComponents, standardize=True):
    from sklearn.decomposition import PCA, NMF

    if standardize:
        newX = np.reshape(X, (-1, X.shape[2]))
        scaler = preprocessing.StandardScaler().fit(newX)  
        newX = scaler.transform(newX)
        X = np.reshape(newX, (X.shape[0],X.shape[1],X.shape[2]))
    
    newX = np.reshape(X, (-1, X.shape[2]))

    if compression == "PCA":
        feature_extraction = PCA(n_components=numComponents, whiten=True)
    elif compression == "NMF":
        feature_extraction = NMF(n_components=numComponents)
    else:
        raise ValueError("Unknown compression method "+compression)

    newX = feature_extraction.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, feature_extraction

def shuffle_train_test(train, test):
    np.random.seed(41)
    for i in range(train.shape[0]):
        for j in range(train.shape[1]):
            if train[i, j] != 0 or test[i, j] != 0 : #eviter calcul inutiles
                x = np.random.randint(1,3)
                if x == 1:
                    temp = train[i, j]
                    train[i, j] = test[i, j]
                    test[i, j] = temp
    return train, test

def delete_useless_classes(data, classes_authorized):
    #data = data[data.any() in classes_authorized]
    #if not data
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j] not in classes_authorized:
                data[i][j] = 0
            if data[i][j] == 2:
                data[i][j] = 1
            if data[i][j] == 3:
                data[i][j] = 2
            if data[i][j] == 5:
                data[i][j] = 3
            if data[i][j] == 6:
                data[i][j] = 4
            if data[i][j] == 10:
                data[i][j] = 5
            if data[i][j] == 11:
                data[i][j] = 6
            if data[i][j] == 12:
                data[i][j] = 7
            if data[i][j] == 14:
                data[i][j] = 8
            if data[i][j] == 15:
                data[i][j] = 9
    return data

def preprocess_dataset(classes_authorized, components, compression_method, patch_size):
    X, train_data, test_data = pp.load_data()

    train_data = pp.delete_useless_classes(train_data, classes_authorized)
    test_data = pp.delete_useless_classes(test_data, classes_authorized)
    print("Before Shuffle: ")
    pretty_print_count(train_data, test_data)
    train_data, test_data = pp.shuffle_train_test(train_data, test_data)
    print("After Shuffle: ")
    pretty_print_count(train_data, test_data)

    if compression_method is not None:
        X, pca = pp.dimensionality_reduction(X, numComponents=components, standardize=False,
                                             compression=compression_method)

    # CREATE PATCHES, DELETE 0 VALUES
    X_train, X_test, y_train, y_test = pp.patch_1dim_split(X, train_data, test_data, patch_size)

    y_train = np_utils.to_categorical(y_train, num_classes=9)
    y_test = np_utils.to_categorical(y_test, num_classes=9)

    t, v = np.unique(train_data, return_counts=True)
    print(t, v)
    t, v = np.unique(test_data, return_counts=True)
    print(t, v)

    return X, X_train, X_test, y_train, y_test

# Source: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def print_progress_bar(iteration, total, prefix ='Progress: ', suffix =' Complete', decimals = 1, length = 40, fill ='█'):
    """
    Call in a loop to create terminal progress bar
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    sys.stdout.flush()
    if iteration == total:
        print()