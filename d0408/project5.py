from skimage.feature import hog
from sklearn.datasets import fetch_lfw_people
from sklearn import svm
from sklearn.model_selection import train_test_split
import cv2
import numpy as np


def get_clf_result(kernel_: str, x_train, x_test, y_train, y_test, c: int, gamma_: str) -> float:
    # create model
    clf = svm.SVC(kernel=kernel_, C=c, gamma=gamma_)

    # train model
    clf.fit(x_train, y_train)

    # get the rate of score
    return round(clf.score(x_test, y_test), 4)


def run():
    data_amount = 487

    lfw_people = fetch_lfw_people(min_faces_per_person=200, resize=0.4)

    car = list()
    for x in range(data_amount):
        img = cv2.imread('./d0408/assets/car/%05d.jpg' % (x + 1))
        img = cv2.resize(img, (37, 50))
        car.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    car = np.array(car)

    image_people = lfw_people['images'][:data_amount]
    for p in range(len(image_people)):
        image_people[p] = cv2.resize(image_people[p], (37, 50))

    target_people = [0] * data_amount
    target_car = [1] * data_amount
    target = target_car + target_people

    images = list()
    for x in car:
        images.append(x)

    for x in image_people:
        images.append(x)

    hogged_images = []
    for image in images:
        fd, hog_image = hog(
            image,
            orientations=8,
            pixels_per_cell=(9, 9),
            cells_per_block=(1, 1),
            visualize=True,
        )
        hogged_images.append(fd)

    car_test = list()
    for x in range(10):
        img = cv2.imread('./d0408/assets/test_data/%05d.jpg' % (x + 1))
        img = cv2.resize(img, (37, 50))
        car_test[x] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    avatar_of_professors_in_ntut_cs = list()
    for x in range(10):
        img = cv2.imread('./d0408/assets/test_data/%d.jpg' % (x + 1))
        img = cv2.resize(img, (37, 50))
        avatar_of_professors_in_ntut_cs[x] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # split some data to be the test data
    x_train, x_test, y_train, y_test = train_test_split(
        hogged_images, target, test_size=0.2, random_state=0
    )

    target_test = ([1] * 10) + ([0] * 10)
    image_test = car_test + avatar_of_professors_in_ntut_cs

    print(image_test)

    print('Accuracy =', get_clf_result('linear', x_train, image_test, y_train, target_test, 1, 'auto'))
