import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

def load_dataset(folder_path, file_name):
    dataset = pd.read_csv(folder_path + file_name)

    X = dataset['text']
    y = dataset['label']

    X = X.values.astype(str)
    y = y.values.astype('int')

    return X, y

def print_accuracy_measures(test_y, pred_y):
    print("Accuracy of Logistic Regression Classifier: {}%".format(round(accuracy_score(test_y, pred_y) * 100, 2)))
    print("\nConfusion Matrix of Logistic Regression Classifier:\n")
    print(confusion_matrix(test_y, pred_y))
    print("\nCLassification Report of Logistic Regression Classifier:\n")
    print(classification_report(test_y, pred_y))