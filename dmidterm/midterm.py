import numpy as np
import cv2
from sklearn import svm
import os
from sklearn.model_selection import train_test_split

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


def run():
    # get data for training.
    train_paper_data = convert_to_gray(get_data('train', 'paper'))
    train_rock_data = convert_to_gray(get_data('train', 'rock'))
    train_scissors_data = convert_to_gray(get_data('train', 'rock'))

    # get data for testing.
    test_paper_data = convert_to_gray(get_data('test', 'paper'))
    test_rock_data = convert_to_gray(get_data('test', 'rock'))
    test_scissors_data = convert_to_gray(get_data('test', 'rock'))

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
        np.array(train_data + test_data),
        np.array(train_target + test_target),
        test_size=0.2, random_state=0
    )

    # train
    print(get_clf_result('linear', *result, 1, 'auto'))
