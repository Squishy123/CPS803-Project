import util
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit

# References
# https://analyticsindiamag.com/hands-on-guide-to-predict-fake-news-using-logistic-regression-svm-and-naive-bayes-methods/
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

def main(folder_path, train_valid_file, test_file, save_file):
    X, y = util.load_dataset(folder_path, train_valid_file)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)

    title = "Logistic Regression Model"
    estimator = LogisticRegression()

    cv = ShuffleSplit(n_splits=10, test_size=0.1275, random_state=0)

    plot = util.plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=4)
    plot.show()
    plot.savefig('log_reg_default_learning_curve.png')


class LogisticRegressionModel:
    """
    Example usage:
        > clf = LogisticRegressionModel()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self):
        """
        Args:
        """
        self.model = None

    def fit(self, X, y, ngram=(1, 1)):
        pl = Pipeline([('tfidf', TfidfVectorizer(stop_words=text.ENGLISH_STOP_WORDS, ngram_range=ngram)),
                       ('model', LogisticRegression())])
        self.model = pl.fit(X, y)

    def predict(self, X):
        lr_pred = self.model.predict(X)

        return lr_pred

if __name__ == "__main__":
    folder_path = 'datasets/kaggle_clement/split_files/'
    train_valid_file = 'train_valid.csv'
    test_file = 'test.csv'
    save_file = 'test_pred.csv'

    main(folder_path, train_valid_file, test_file, save_file)
