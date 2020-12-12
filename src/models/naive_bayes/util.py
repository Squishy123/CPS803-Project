import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report
from sklearn.feature_extraction import text
from pathlib import Path
from wordcloud import WordCloud

def load_dataset(folder_path, file_name):
    dataset = pd.read_csv(folder_path + file_name)

    X = dataset['text']
    y = dataset['label']

    X = X.values.astype(str)
    y = y.values.astype('int')

    return X, y

def visualize_confusion_matrix(estimator, X, Y, title="Model Confusion Matrix", savefig=True):
    disp = plot_confusion_matrix(estimator,X, Y, cmap=plt.cm.Blues)
    disp.ax_.set_title(title)

    if(savefig):
        plt.savefig("plots/"+title+".jpg")

    return disp

def visualize_k_folds(classes, title, cv, savefig=".jpg"):

    return 

def plot_learning_curve(estimator, title, X, y, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), savefig=True):
    _, axes = plt.subplots(1, 1)
    axes.set_title(title)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes.legend(loc="best")

    if(savefig):
        plt.savefig("plots/"+title+".jpg")

    return plt

def plot_cv_score(cv_results, title="CV Score", savefig=True):
    x=np.arange(len(cv_results["test_score"]))
    y=cv_results["test_score"]*100

    fig,ax = plt.subplots()
    plt.ylim(np.min(y)-10,np.min([100, np.max(y)+10]))
    plt.bar(x,y)
    plt.ylabel("Accuracy Score")
    ax.set_title(title)
    plt.xticks(x, ("Fold " + str(i+1) for i in x))

    if(savefig):
        plt.savefig("plots/"+title+".jpg")

    return plot_cv_score

def print_accuracy_measures(test_y, pred_y, label="Logistic Regression", saveFile=True):
    str="Accuracy of " + label + " Classifier: {}%".format(round(accuracy_score(test_y, pred_y) * 100, 2))
    str+="\nCLassification Report of " + label +  " Classifier:\n"
    str+=classification_report(test_y, pred_y)

    print(str)

    if saveFile:
        p = Path('plots')
        p.mkdir(exist_ok=True)
        f=open('plots/'+label+"_classification_report.txt","w")
        f.write(str)
        f.close()

def plot_word_cloud(model, save_file="word_cloud"):
    tfidf_freq = model.named_steps['features'].vocabulary_
    coef = list(model.named_steps['model'].coef_[0])

    for word, idx in tfidf_freq.items():
        tfidf_freq[word] = coef[idx]

    wordcloud = WordCloud(width=800, height=800,
                          background_color="white",
                          stopwords=text.ENGLISH_STOP_WORDS,
                          min_font_size=10,
                          min_word_length=2,
                          repeat=False).generate_from_frequencies(tfidf_freq)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig("plots/"+save_file+".jpg")