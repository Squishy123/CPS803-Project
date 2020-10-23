import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

# References
# https://analyticsindiamag.com/hands-on-guide-to-predict-fake-news-using-logistic-regression-svm-and-naive-bayes-methods/

def main():

    train_data = pd.read_csv('../datasets/train1.csv')
    test_data = pd.read_csv('../datasets/test1.csv')

    train_X = train_data['text']
    train_y = train_data['label']

    train_X = train_X.values.astype(str)
    train_y = train_y.values.astype('int')

    test_X = test_data['text']
    test_y = test_data['label']

    test_X = test_X.values.astype(str)
    test_y = test_y.values.astype('int')

    pl = Pipeline([('count_vec', CountVectorizer()), ('tfidf', TfidfTransformer()), ('model', LogisticRegression())])
    lr_model = pl.fit(train_X, train_y)
    lr_pred = lr_model.predict(test_X)

    # understand what each of this means
    print("Accuracy of Logistic Regression Classifier: {}%".format(round(accuracy_score(test_y, lr_pred) * 100, 2)))
    print("\nConfusion Matrix of Logistic Regression Classifier:\n")
    print(confusion_matrix(test_y, lr_pred))
    print("\nCLassification Report of Logistic Regression Classifier:\n")
    print(classification_report(test_y, lr_pred))

if __name__ == "__main__":
    main()