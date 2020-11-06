import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix

def load_dataset(folder_path, file_name):
    dataset = pd.read_csv(folder_path + file_name)

    X = dataset['text']
    y = dataset['label']

    X = X.values.astype(str)
    y = y.values.astype('int')

    return X, y

def plot_conf_mat(y, y_pred):
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    plot = plot_confusion_matrix(conf_mat=cm, show_absolute=True, show_normed=True, colorbar=True)
    return plot

def print_accuracy_measures(test_y, pred_y):
    print("Accuracy of Logistic Regression Classifier: {}%".format(round(accuracy_score(test_y, pred_y) * 100, 2)))

    # repetitive so taking out
    # print("\nConfusion Matrix of Logistic Regression Classifier:\n")
    # print(confusion_matrix(test_y, pred_y))

    print("\nCLassification Report of Logistic Regression Classifier:\n")
    print(classification_report(test_y, pred_y))