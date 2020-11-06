from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline
import util

folder_path = '../datasets/'
train_file = 'train1.csv'
test_file = 'test1.csv'
save_file = 'pred1.csv'
train_X, train_Y = util.load_dataset(folder_path, train_file)
test_X, test_Y = util.load_dataset(folder_path, test_file)

model = make_pipeline(TfidfVectorizer(), BernoulliNB())
model.fit(train_X, train_Y)

pred_Y = model.predict(test_X)
util.print_accuracy_measures(test_Y, pred_Y)