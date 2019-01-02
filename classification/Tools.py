import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def iterated_benchmark_classifier(clf, x, y, num_tests=100, test_size=0.3):
    """
    Fit and Predict score num_test times to get signifiacant metrics
    :param clf: Classifier instance
    :param x: List of descriptors
    :param y: List of labels
    :param num_tests: Number of times that the classifier'll be fitted and scored
    :return: Score benchmark metrics
    """
    scores = []
    labels = ['no-educacion', 'educacion']
    for _ in range(num_tests):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        ret = classification_report(
            y_test,
            y_pred,
            target_names=labels,
            output_dict=True
        )
        scores.append(ret)

    precision = [[] for _ in labels]
    recall = [[] for _ in labels]
    f1 = [[] for _ in labels]
    support = [[] for _ in labels]
    for score in scores:
        for idx, label in enumerate(labels):
            precision[idx].append(score[label]['precision'])
            recall[idx].append(score[label]['recall'])
            f1[idx].append(score[label]['f1-score'])
            support[idx].append(score[label]['support'])

    mean_precision = np.mean(precision, axis=1)
    mean_recall = np.mean(recall, axis=1)
    mean_f1 = np.mean(f1, axis=1)
    mean_support = np.mean(support, axis=1)

    dic = {}
    for idx, label in enumerate(labels):
        dic[label] =  {
            'precision': mean_precision[idx],
            'recall': mean_recall[idx],
            'f1': mean_f1[idx],
            'support': mean_support[idx]
        }
    return dic


def benchmark_classifier(clf, x, y):
    """
    Fit and predict for metrics.
    :param clf: Classifier instance
    :param x: List of descriptors
    :param y: List of labels
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    ret = classification_report(
        y_test,
        y_pred,
        target_names=["no-educacion", "educacion"],
        output_dict=True
    )
    return ret


def get_classifier_benchmarks(x, y, model_name):
    classifiers = [
        ("Base", DummyClassifier(strategy='stratified')),
        ("SVM", SVC(kernel='linear')),
        ("DT", DecisionTreeClassifier()),
        ("NB", GaussianNB()),
        ("KNN", KNeighborsClassifier(n_neighbors=5)),
        ("GP", GaussianProcessClassifier()),
        ("MLP", MLPClassifier())
    ]
    iter = 300
    print("info: benchmarking descriptor model: {} with {} classifiers {} iterations".format(model_name, len(classifiers), iter))
    metrics = {}
    for name, clf in classifiers:
        print("info: benchmarking {}".format(name))
        metrics[name] = iterated_benchmark_classifier(clf, x, y, num_tests=iter, test_size=0.2)
    return metrics
