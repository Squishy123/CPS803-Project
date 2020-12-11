import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report
from sklearn.feature_extraction import text
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
        plt.savefig(title+".png")

    return disp

def visualize_k_folds(classes, title, cv, savefig=".png"):

    return 

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
 
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

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
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    '''
    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")
    '''

    return plt

def plot_word_cloud(model, save_file="word_cloud.png"):
    tfidf_freq = model.named_steps['tfidf'].vocabulary_
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
    plt.savefig(save_file)

def print_accuracy_measures(test_y, pred_y, label="Logistic Regression"):
    print("Accuracy of Logistic Regression Classifier: {}%".format(round(accuracy_score(test_y, pred_y) * 100, 2)))

    print("\nCLassification Report of " + label +  " Classifier:\n")
    print(classification_report(test_y, pred_y))