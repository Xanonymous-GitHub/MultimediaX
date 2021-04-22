import numpy as np
import cv2
from sklearn import svm
import os
from sklearn.model_selection import train_test_split
from scipy.cluster.vq import *

path_prefix = './dmidterm/assets/'


def show_img(*images):
    if not isinstance(images, list):
        images = [images]

    for image in images[0]:
        cv2.imshow('', image)
        cv2.waitKey(0)


def get_clf_result(kernel_: str, x_train, x_test, y_train, y_test, c: int, gamma_: str) -> float:
    # create model
    clf = svm.SVC(kernel=kernel_, C=c, gamma=gamma_)

    # train model
    clf.fit(x_train, y_train)

    # get the rate of score
    return round(clf.score(x_test, y_test), 3)


def get_data(from_: str, hand_type: str) -> list:
    # get all file name in target dir.
    whole_path = path_prefix + from_ + '/' + hand_type + '/'
    names = [f for f in os.listdir(whole_path)]
    return [cv2.imread(whole_path + name) for name in names]


def convert_to_gray(images: list) -> list:
    return [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]


def create_target_list(amount: int, content: str) -> list:
    return [content for _ in range(amount)]


def get_attributes_from_images(images: list):
    sift = cv2.SIFT_create()
    data_size = len(images)
    des_list = []
    for data in images:
        kpts = sift.detect(data)
        _, des = sift.compute(data, kpts)
        des_list.append(des)

    descriptors = des_list[0]
    for descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    k_means = 60
    voc, variance = kmeans(descriptors, k_means, 1)

    im_features = np.zeros((data_size, k_means), 'float32')
    for i in range(data_size):
        words, distance = vq(des_list[i], voc)
        for word in words:
            im_features[i][word] += 1

    return im_features


def run():
    # get data for training.
    train_paper_data = convert_to_gray(get_data('train', 'paper'))
    train_rock_data = convert_to_gray(get_data('train', 'rock'))
    train_scissors_data = convert_to_gray(get_data('train', 'scissors'))

    # get data for testing.
    test_paper_data = convert_to_gray(get_data('test', 'paper'))
    test_rock_data = convert_to_gray(get_data('test', 'rock'))
    test_scissors_data = convert_to_gray(get_data('test', 'scissors'))

    # mark targets related to datasets
    train_paper_target = create_target_list(len(train_paper_data), 'paper')
    train_rock_target = create_target_list(len(train_rock_data), 'rock')
    train_scissors_target = create_target_list(len(train_scissors_data), 'scissors')
    test_paper_target = create_target_list(len(test_paper_data), 'paper')
    test_rock_target = create_target_list(len(test_rock_data), 'rock')
    test_scissors_target = create_target_list(len(test_scissors_data), 'scissors')

    # combine train data
    train_data = train_paper_data + train_rock_data + train_scissors_data

    # combine train target
    train_target = train_paper_target + train_rock_target + train_scissors_target

    # combine test data
    test_data = test_paper_data + test_rock_data + test_scissors_data

    # combine test target
    test_target = test_paper_target + test_rock_target + test_scissors_target

    # split data
    result = train_test_split(
        np.concatenate((get_attributes_from_images(train_data), get_attributes_from_images(test_data))),
        np.array(train_target + test_target),
        test_size=0.2, random_state=0
    )

    # train
    print(get_clf_result('linear', *result, 1, 'auto'))
