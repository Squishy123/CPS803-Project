from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.linear_model import LogisticRegression

def LogRegModel(ngram=(1, 1)):
    return Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=text.ENGLISH_STOP_WORDS, ngram_range=ngram)),
        ('model', LogisticRegression())
    ])