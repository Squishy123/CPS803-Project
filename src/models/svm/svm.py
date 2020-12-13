from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import text
from sklearn import svm

def BaseSVMModel(ngram=(1, 1)):
    return Pipeline([
        ('features', CountVectorizer(max_features=1000, stop_words=text.ENGLISH_STOP_WORDS, ngram_range=ngram)),
        ('model', svm.SVC(kernel="linear", C=0.025))])

def TFIDFSVMModel(ngram=(1, 1)):
    return Pipeline([
        ('features', TfidfVectorizer(max_features=1000, stop_words=text.ENGLISH_STOP_WORDS, ngram_range=ngram)),
        ('model', svm.SVC(kernel="linear", C=0.025))])