from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def run_1():
    iris = datasets.load_iris()

    x, y = iris.data, iris.target
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=4)

    k_range = range(1, 26)
    scores = {}
    scores_list = []

    # knn
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        scores[k] = metrics.accuracy_score(y_test, y_pred)
        scores_list.append(metrics.accuracy_score(y_test, y_pred))

    plt.plot(k_range, scores_list)
    plt.show()
