import numpy as np
import cv2
from sklearn import svm
import os
import random
import joblib
from sklearn.model_selection import train_test_split
from scipy.cluster.vq import *

path_prefix = './dmidterm/assets/'
model_path = './dmidterm/model.pkl'


def show_img(*images):
    if not isinstance(images, list):
        images = [images]

    for image in images[0]:
        cv2.imshow('', image)
        cv2.waitKey(0)


def get_clf_model(kernel_: str, x_train, y_train, c: int, gamma_: str):
    # create model
    clf = svm.SVC(kernel=kernel_, C=c, gamma=gamma_)

    # train model
    clf.fit(x_train, y_train)

    # get the trained model
    return clf


def get_data(from_: str, hand_type: str) -> list:
    # get all file name in target dir.
    whole_path = path_prefix + from_ + '/' + hand_type + '/'
    names = [f for f in os.listdir(whole_path)]
    return [cv2.imread(whole_path + name) for name in names]


def get_single_validation_img(hand_type: str):
    whole_path = path_prefix + 'validation/'
    names = list(filter(lambda x: hand_type in x, [f for f in os.listdir(whole_path)]))
    c = random.choice([cv2.imread(whole_path + name) for name in names])
    return convert_to_gray([c])[0]


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

    k_means = 120
    voc, variance = kmeans(descriptors, k_means, 1)

    im_features = np.zeros((data_size, k_means), 'float32')
    for i in range(data_size):
        words, distance = vq(des_list[i], voc)
        for word in words:
            im_features[i][word] += 1

    return im_features


def start_validation(model):
    while True:
        try:
            chosen_hand_type = input('please tell me what is your hand (paper|rock|scissors)?')
        except EOFError:
            break
        if chosen_hand_type not in ['paper', 'rock', 'scissors']:
            print('please provide a correct hand type!\n')
            continue

        random_hand_img = get_single_validation_img(chosen_hand_type)
        img_data = get_attributes_from_images([random_hand_img])[0]
        expected_target = model.predict(img_data)
        print('The expected result is: ' + expected_target)


def run():
    # detect if model trained.
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        model = None

    if model is None:
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
        train_data_origin = train_paper_data + train_rock_data + train_scissors_data

        # combine train target
        train_target_origin = train_paper_target + train_rock_target + train_scissors_target

        # combine test data
        test_data_origin = test_paper_data + test_rock_data + test_scissors_data

        # combine test target
        test_target_origin = test_paper_target + test_rock_target + test_scissors_target

        # SIFT
        train_data = get_attributes_from_images(train_data_origin)
        test_data = get_attributes_from_images(test_data_origin)

        # split data
        # result = train_test_split(
        #     np.concatenate((train_data, test_data)),
        #     np.array(train_target_origin + test_target_origin),
        #     test_size=0.2, random_state=0
        # )

        # get trained model
        model = get_clf_model('linear', train_data, train_target_origin, 1, 'auto')
        joblib.dump(model, model_path)

    start_validation(model)
