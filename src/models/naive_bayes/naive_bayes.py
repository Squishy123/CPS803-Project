import util
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.naive_bayes import MultinomialNB

# def main(folder_path, train_file, test_file, save_file):

    # train_X, train_y = util.load_dataset(folder_path, train_file)
    # test_X, test_y = util.load_dataset(folder_path, test_file)
    #
    # lrm = NaiveBayesModel()
    # lrm.fit(train_X, train_y, (1, 1))
    # pred_y = lrm.predict(test_X)
    #
    # util.print_accuracy_measures(test_y, pred_y)

class NaiveBayesModel:
    """
    Example usage:
        > clf = NaiveBayesModel()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self):
        """
        """
        self.model = None

    def fit(self, X, y, ngram=(1, 1)):
        pl = Pipeline([('tfidf', TfidfVectorizer(stop_words=text.ENGLISH_STOP_WORDS, ngram_range=ngram)),
                       ('nb', MultinomialNB())])
        self.model = pl.fit(X, y)

    def predict(self, X):
        lr_pred = self.model.predict(X)

        return lr_pred

# if __name__ == "__main__":
#     folder_path = 'datasets/kaggle_comp/split_files/'
#     train_file = 'train.csv'
#     test_file = 'test.csv'
#     save_file = 'test_pred.csv'
#
#     main(folder_path, train_file, test_file, save_file)
