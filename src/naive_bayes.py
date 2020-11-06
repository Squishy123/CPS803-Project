from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import plot_confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import util

folder_path = 'datasets/kaggle_clement/'
train_file = 'train.csv'
test_file = 'test.csv'
save_file = 'test_pred.csv'
train_X, train_Y = util.load_dataset(folder_path, train_file)
test_X, test_Y = util.load_dataset(folder_path, test_file)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train_X, train_Y)
pred_Y = model.predict(test_X)

disp = plot_confusion_matrix(model, test_X, test_Y,cmap=plt.cm.Blues)
plt.savefig("naive_bayes_confusion.png")
print("Model Accuracy: " + str(round(accuracy_score(test_Y, pred_Y) * 100, 2)) + "%")
print(classification_report(test_Y,pred_Y))







