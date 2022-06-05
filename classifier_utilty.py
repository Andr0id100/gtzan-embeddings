from sklearn import linear_model
import warnings
from sklearn.exceptions import ConvergenceWarning


def create_classifier():
    classifier = linear_model.LogisticRegression()
    return classifier


def train_classifier(classifier, train_embeddings, train_labels):
    # Supress warning about convergance
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        classifier.fit(train_embeddings, train_labels)


def test_classifier(classifier, test_embeddings, test_labels):
    print(f"Accuracy:", classifier.score(test_embeddings, test_labels))
