import util
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.linear_model import LogisticRegression

# References
# https://analyticsindiamag.com/hands-on-guide-to-predict-fake-news-using-logistic-regression-svm-and-naive-bayes-methods/

def main(folder_path, train_file, test_file, save_file):

    test = [
        [
            ["This is a sentence"],
            ["The sky is blue and white"],
            ["I don't want to study"]
        ],
        [
            ["Can dogs see colours"],
            ["What goes on in cat's minds"]
        ],
        [
            ["Spaghetti is made by boiling noodles"],
            ["Use fresh tomatoes for better flavour"],
            ["Don't forget the garlic bread"]
        ]
    ]

    test2 = [
         "This is a sentence. The sky is blue and white. I don't want to study",
        "Can dogs see colours? What goes on in cat's minds",
        "Spaghetti is made by boiling noodles. Use fresh tomatoes for better flavour. Don't forget the garlic bread"
    ]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(test2)
    print(vectorizer.get_feature_names())

    # train_X, train_y = util.load_dataset(folder_path, train_file)
    # test_X, test_y = util.load_dataset(folder_path, test_file)

    # lrm = LogisticRegressionModel()
    # lrm.fit(train_X, train_y, (1, 2))
    # pred_y = lrm.predict(test_X)
    #
    # util.print_accuracy_measures(test_y, pred_y)

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
    folder_path = 'datasets/kaggle_ruchi/split_files/'
    train_file = 'train.csv'
    test_file = 'test.csv'
    save_file = 'test_pred.csv'

    main(folder_path, train_file, test_file, save_file)
