import matplotlib.pyplot as plt
from skimage.feature import hog

from sklearn.datasets import fetch_lfw_people, fetch_california_housing
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split


def get_clf_result(kernel_: str, x_train, x_test, y_train, y_test, c: int, gamma_: str) -> float:
    # create model
    clf = svm.SVC(kernel=kernel_, C=c, gamma=gamma_)

    # train model
    clf.fit(x_train, y_train)

    # get the rate of score
    return round(clf.score(x_test, y_test), 3)


def run():
    # imgs_to_use = ['camera', 'text', 'coins', 'moon',
    #                'page', 'clock', 'immunohistochemistry',
    #                'chelsea', 'coffee', 'hubble_deep_field']

    lfw_people = fetch_lfw_people()
    california_housing = fetch_california_housing()

    # images_not_people = [color.rgb2gray(getattr(data, name)())
    #                      for name in imgs_to_use]
    image_people = california_housing['data'][:13233]
    image_house = california_housing['data'][:13233]

    hogged_house = []
    for image in image_house:
        fd, hog_image = hog(
            image,
            orientations=8,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
            visualize=True,
        )
        hogged_house.append(hog_image[0])

    target_house = [1] * len(image_house)

    print(len(image_house))

    hogged_people = []
    for image in image_people:
        fd, hog_image = hog(
            image,
            orientations=8,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
            visualize=True,
        )
        hogged_people.append(hog_image[0])

    target_people = [0] * len(image_people)

    print(len(image_people))

    # split some data to be the test data
    ml_data = train_test_split(
        hogged_house + hogged_people, target_house + target_people, test_size=0.2, random_state=0
    )

    plt.imshow(image_house[0])
    plt.imshow(hogged_house[0])
    plt.show()

    # print('linear', ' c=', 1, 'gamma=auto', get_clf_result('linear', *ml_data, 1, 'auto'))
