from sklearn import datasets, svm
from sklearn.model_selection import train_test_split


def get_clf_result(kernel_: str, x_train, x_test, y_train, y_test, c: int, gamma_: str) -> float:
    clf = svm.SVC(kernel=kernel_, C=c, gamma=gamma_)
    clf.fit(x_train, y_train)
    return round(clf.score(x_test, y_test), 3)


def run_2():
    wine = datasets.load_wine()

    x, y = wine.data, wine.target

    data = train_test_split(
        x, y, test_size=0.2, random_state=4
    )

    for c in range(1, 11):
        print('linear', 'c=', c, get_clf_result('linear', *data, c, 'auto'))
        print('poly', 'c=', c, get_clf_result('poly', *data, c, 'auto'))
        print('rbf', 'c=', c, get_clf_result('rbf', *data, c, 'auto'))
        print('sigmoid', 'c=', c, get_clf_result('sigmoid', *data, c, 'auto'))
        print()
