# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# <h1> Logistic Regression Model on Kaggle Competition Fake News Dataset </h1>
# <h3> The fake news dataset is publicly available <a href="https://www.kaggle.com/c/fake-news">here</a></h3>

# %%
import util
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import accuracy_score, plot_confusion_matrix
import log_reg as lr


# %%
folder_path = '../../datasets/kaggle_comp/split_files/'
train_file = 'train.csv'
test_file = 'test.csv'

train_X, train_Y = util.load_dataset(folder_path, train_file)
test_X, test_Y = util.load_dataset(folder_path, test_file)

# %% [markdown]
# <h2> Base Model with Term Frequency</h2>

# %%
base_model = lr.BaseLogRegModel()
base_model.fit(train_X, train_Y)


# %%
pred_Y = base_model.predict(test_X)
util.print_accuracy_measures(test_Y, pred_Y, label="log_reg_uni_base_comp")


# %%
util.visualize_confusion_matrix(base_model,test_X, test_Y,"log_reg_uni_base_comp_confusion_matrix")


# %%
base_cv_results = cross_validate(base_model, train_X, train_Y, cv=KFold(5))
util.plot_cv_score(base_cv_results,title="log_reg_uni_base_comp_cv_score_bar")


# %%
util.plot_learning_curve(base_model, "log_reg_uni_base_comp_learning_curve", train_X,train_Y, cv=KFold(5), n_jobs=4)


# %%
util.plot_word_cloud(base_model,"log_reg_uni_base_comp_word_cloud")

# %% [markdown]
# <h2> Adding TFIDF </h2>

# %%
tfidf_model = lr.TFIDFLogRegModel()
tfidf_model.fit(train_X, train_Y)


# %%
pred_Y = tfidf_model.predict(test_X)
util.print_accuracy_measures(test_Y, pred_Y, label="log_reg_uni_tfidf_comp")


# %%
util.visualize_confusion_matrix(base_model,test_X, test_Y,"log_reg_uni_tfidf_comp_confusion_matrix")


# %%
tfidf_cv_results = cross_validate(base_model, train_X, train_Y, cv=KFold(5))
util.plot_cv_score(base_cv_results,title="log_reg_uni_tfidf_comp_cv_score_bar")


# %%
util.plot_learning_curve(base_model, "log_reg_uni_tfidf_comp_learning_curve", train_X,train_Y, cv=KFold(5), n_jobs=4)


# %%
util.plot_word_cloud(tfidf_model,"log_reg_uni_tfidf_comp_word_cloud")


