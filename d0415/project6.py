import cv2
# import imutils
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from scipy.cluster.vq import *


def get_images(amount: int, data_type: str):
    image_list = list()

    for x in range(1, amount + 1):
        if data_type == 'cat':
            img = cv2.imread('./d0415/assets/training_set/cats/cat.%d.jpg' % x)
        elif data_type == 'car':
            img = cv2.imread('./d0408/assets/car/%05d.jpg' % x)
        else:
            return
        image_list.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return image_list


def get_clf_result(x_train, x_test, y_train, y_test) -> float:
    # create model
    clf = LinearSVC()

    # train model
    clf.fit(x_train, y_train)

    # get the rate of score
    return round(clf.score(x_test, y_test), 4)


def run():
    cat_amount = 200
    car_amount = 200
    data_size = cat_amount + car_amount

    # cat 0 car 1
    ml_data = get_images(cat_amount, 'cat') + get_images(car_amount, 'car')
    ml_target = [0] * cat_amount + [1] * car_amount

    sift = cv2.SIFT_create()
    des_list = []
    for data in ml_data:
        data = cv2.resize(data, (300, 300))
        kpts = sift.detect(data)
        _, des = sift.compute(data, kpts)
        des_list.append(des)

    # print(des_list)

    descriptors = des_list[0]
    for descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    k_means = 1000
    voc, variance = kmeans(descriptors, k_means, 1)

    im_features = np.zeros((data_size, k_means), 'float32')
    for i in range(data_size):
        words, distance = vq(des_list[i], voc)
        for word in words:
            im_features[i][word] += 1

    print('data_size =', data_size)

    result = train_test_split(im_features, ml_target, test_size=0.2, random_state=0)
    print(get_clf_result(*result))
