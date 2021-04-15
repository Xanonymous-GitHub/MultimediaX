import cv2
from sklearn import svm


def get_images(amount: int, data_type: str):
    image_list = list()
    path = './d0415/assets/'
    for x in range(1, amount + 1):
        if data_type == 'train_cat':
            img = cv2.imread(path + 'training_set/cats/cat.%d.jpg' % x)
        elif data_type == 'train_dog':
            img = cv2.imread(path + 'training_set/dogs/dog.%d.jpg' % x)
        elif data_type == 'test_cat':
            img = cv2.imread(path + 'test_set/cats/cat.%d.jpg' % x)
        elif data_type == 'test_dog':
            img = cv2.imread(path + 'test_set/dogs/dog.%d.jpg' % x)
        else:
            return

        image_list.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return image_list


def get_clf_result(kernel_: str, x_train, x_test, y_train, y_test, c: int, gamma_: str) -> float:
    # create model
    clf = svm.SVC(kernel=kernel_, C=c, gamma=gamma_)

    # train model
    clf.fit(x_train, y_train)

    # get the rate of score
    return round(clf.score(x_test, y_test), 4)


def run():
    train_amount = 500
    test_amount = 10
