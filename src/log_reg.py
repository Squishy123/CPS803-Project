import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

# References
# https://analyticsindiamag.com/hands-on-guide-to-predict-fake-news-using-logistic-regression-svm-and-naive-bayes-methods/

def main():

    train_data = pd.read_csv('../datasets/train1.csv')
    test_data = pd.read_csv('../datasets/test1.csv')

    train_X = train_data.iloc[:, :-1]
    train_y = train_data.iloc[:, -1]

    # pl = Pipeline([('count_vec', CountVectorizer()), ('tfidf', TfidfTransformer()), ('model', LogisticRegression())])
    # lr_model = pl.fit(train_X, train_y)
    # error Found input variables with inconsistent numbers of samples: [3, 16640]

if __name__ == "__main__":
    main()