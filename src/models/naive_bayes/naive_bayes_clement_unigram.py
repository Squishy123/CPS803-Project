# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# <h1> Naive Bayes Model on Kaggle Clement Fake News Dataset </h1>
# <h3> The fake news dataset is publicly available <a href="https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset">here</a></h3>

# %%
import util
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import accuracy_score, plot_confusion_matrix
import naive_bayes as nb


# %%
folder_path = '../../datasets/kaggle_clement/split_files/'
train_file = 'train.csv'
test_file = 'test.csv'

train_X, train_Y = util.load_dataset(folder_path, train_file)
test_X, test_Y = util.load_dataset(folder_path, test_file)

# %% [markdown]
# <h2> Base Model with Term Frequency</h2>

# %%
base_model = nb.BaseNaiveBayesModel()
base_model.fit(train_X, train_Y)


# %%
pred_Y = base_model.predict(test_X)
util.print_accuracy_measures(test_Y, pred_Y, label="naive_bayes_uni_base_clement")


# %%
util.visualize_confusion_matrix(base_model,test_X, test_Y,"naive_bayes_uni_base_clement_confusion_matrix")


# %%
base_cv_results = cross_validate(base_model, train_X, train_Y, cv=KFold(5))
util.plot_cv_score(base_cv_results,title="naive_bayes_uni_base_clement_cv_score_bar")


# %%
util.plot_learning_curve(base_model, "naive_bayes_uni_base_clement_learning_curve", train_X,train_Y, cv=KFold(5), n_jobs=4)


# %%
util.plot_word_cloud(base_model,"naive_bayes_uni_base_clement_word_cloud")

# %% [markdown]
# <h2> Adding TFIDF </h2>

# %%
tfidf_model = nb.TFIDFNaiveBayesModel()
tfidf_model.fit(train_X, train_Y)


# %%
pred_Y = tfidf_model.predict(test_X)
util.print_accuracy_measures(test_Y, pred_Y, label="naive_bayes_uni_tfidf_clement")


# %%
util.visualize_confusion_matrix(tfidf_model,test_X, test_Y,"naive_bayes_uni_tfidf_clement_confusion_matrix")


# %%
tfidf_cv_results = cross_validate(tfidf_model, train_X, train_Y, cv=KFold(5))
util.plot_cv_score(tfidf_cv_results,title="naive_bayes_uni_tfidf_clement_cv_score_bar")


# %%
util.plot_learning_curve(tfidf_model, "naive_bayes_uni_tfidf_clement_learning_curve", train_X,train_Y, cv=KFold(5), n_jobs=4)


# %%
util.plot_word_cloud(tfidf_model,"naive_bayes_uni_tfidf_clement_word_cloud")


