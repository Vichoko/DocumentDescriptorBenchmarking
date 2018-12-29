import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def iterated_benchmark_classifier(clf, x, y, num_tests=100):
    """
    Fit and Predict score num_test times to get signifiacant metrics
    :param clf: Classifier instance
    :param x: List of descriptors
    :param y: List of labels
    :param num_tests: Number of times that the classifier'll be fitted and scored
    :return: Score benchmark metrics
    """
    scores = []
    for _ in range(num_tests):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        clf.fit(x_train, y_train)
        scores.append(clf.score(x_test, y_test))
    return np.asarray(scores)


def benchmark_classifier(clf, x,y):
    """
    Fit and predict for metrics.
    :param clf: Classifier instance
    :param x: List of descriptors
    :param y: List of labels
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    clf.fit(x_train, y_train)
    return classification_report(y_test, clf.predict(x_test))


def get_classifier_benchmarks(x, y, model_name):
    classifiers = [
        ("Base", DummyClassifier(strategy='stratified')),
        ("SVM", SVC(kernel='linear')),
        ("DT", DecisionTreeClassifier()),
        ("NB", GaussianNB()),
        ("KNN", KNeighborsClassifier(n_neighbors=5))
    ]
    print("info: benchmarking descriptor model: {} with many classifiers".format(model_name))
    for name, clf in classifiers:
        print("Classifier: {}".format(name))
        metrics = benchmark_classifier(clf, x, y)
        print(metrics)