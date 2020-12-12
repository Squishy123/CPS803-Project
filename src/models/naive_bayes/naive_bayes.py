from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import text
from sklearn.naive_bayes import MultinomialNB

def BaseNaiveBayesModel(ngram=(1, 1)):
    return Pipeline([
        ('features', CountVectorizer(stop_words=text.ENGLISH_STOP_WORDS, ngram_range=ngram)),
        ('model', MultinomialNB())])

def TFIDFNaiveBayesModel(ngram=(1, 1)):
    return Pipeline([
        ('features', TfidfVectorizer(stop_words=text.ENGLISH_STOP_WORDS, ngram_range=ngram)),
        ('model', MultinomialNB())])