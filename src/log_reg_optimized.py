# To add a new cell, type ''
# To add a new markdown cell, type ' [markdown]'

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import plot_confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import util
import time

folder_path = 'datasets/kaggle_clement/'
train_file = 'train.csv'
test_file = 'test.csv'
save_file = 'test_pred.csv'
train_X, train_Y = util.load_dataset(folder_path, train_file)
test_X, test_Y = util.load_dataset(folder_path, test_file)

model = Pipeline(steps=[('Tfidf',TfidfVectorizer()),('log', LogisticRegression())])
model.fit(train_X, train_Y)
pred_Y = model.predict(test_X)

disp = plot_confusion_matrix(model, test_X, test_Y,cmap=plt.cm.Blues)
plt.savefig("log_reg_default_confusion.png")
plt.clf()
print("Model Accuracy: " + str(round(accuracy_score(test_Y, pred_Y) * 100, 2)) + "%")
print(classification_report(test_Y,pred_Y))


dual=[False]
penalty=['l2']
tol=[1e-3,1e-4,1e-5]
max_iter=[80,90,100]

param_grid = {
    'log__dual': dual,
    'log__penalty': penalty,
    'log__tol': tol,
    'log__max_iter': max_iter
}
grid=RandomizedSearchCV(model,param_grid, n_jobs=3, n_iter=10)
start_time = time.time()
grid_result = grid.fit(train_X, train_Y)
print("Execution time: " + str((time.time() - start_time)) + ' ms')
print("Best parameter (CV score=%0.3f):" % grid.best_score_)
print(grid.best_params_)

optimized_pred_Y = grid.best_estimator_.predict(test_X)

disp = plot_confusion_matrix(grid.best_estimator_, test_X, test_Y,cmap=plt.cm.Blues)
plt.savefig("log_reg_optimized_confusion.png")
plt.clf()
print("Model Accuracy: " + str(round(accuracy_score(test_Y, optimized_pred_Y) * 100, 2)) + "%")
print(classification_report(test_Y,optimized_pred_Y))




