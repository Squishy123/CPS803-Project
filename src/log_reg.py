import util
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix

# References
# https://analyticsindiamag.com/hands-on-guide-to-predict-fake-news-using-logistic-regression-svm-and-naive-bayes-methods/

# def main(folder_path, train_file, test_file, save_file):
#
#     train_X, train_y = util.load_dataset(folder_path, train_file)
#     test_X, test_y = util.load_dataset(folder_path, test_file)
#
#     lrm = LogisticRegressionModel()
#     lrm.fit(train_X, train_y)
#     pred_y = lrm.predict(test_X)
#
#     util.print_accuracy_measures(test_y, pred_y)

class LogisticRegressionModel:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.model = None

    def fit(self, X, y):
        pl = Pipeline([('tfidf', TfidfVectorizer()),
                       ('model', LogisticRegression())])
        self.model = pl.fit(X, y)

    def predict(self, X):
        lr_pred = self.model.predict(X)

        return lr_pred

# if __name__ == "__main__":
#     folder_path = 'datasets/kaggle_clement/'
#     train_file = 'train.csv'
#     test_file = 'test.csv'
#     save_file = 'test_pred.csv'
#
#     main(folder_path, train_file, test_file, save_file)
