import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split


def run_2():
    wine = datasets.load_wine()

    x, y = wine.data, wine.target
    print(len(x))
    print(len(y))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=4)

    # set configuration of training model
    clf = svm.SVC(kernel='linear', C=1, gamma='auto')
    # provide data to model and train model
    clf.fit(x_train, y_train)

    print('Predict')
    print(clf.predict(x_train))
    print(clf.predict(x_test))

    print('Accuracy')
    print(clf.score(x_train, y_train))
    print(clf.score(x_test, y_test))
    plt.scatter(x_train[:, 0], x_train[:, 1])
    # plt.plot([0, 1, 3, 4], [1, 3, 11, 14])
    plt.show()
