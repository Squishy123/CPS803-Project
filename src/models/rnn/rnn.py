# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import util


# %%
tf.debugging.set_log_device_placement(True)
# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)

print(tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# %%
def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])


# %%
folder_path = 'datasets/kaggle_clement/split_files/'
train_file = 'train.csv'
test_file = 'test.csv'
save_file = 'test_pred.csv'

train=pd.read_csv(folder_path+train_file, header=0)
train_X,train_Y=(np.array(train["text"]),np.array(train["label"]))

test=pd.read_csv(folder_path+test_file, header=0)
test_X,test_Y=(np.array(train["text"]),np.array(train["label"]))


# %%
encoder = tf.keras.preprocessing.text.Tokenizer()
encoder.fit_on_texts(train_X)
encoded_docs = encoder.texts_to_matrix(train_X, mode='tfidf')


# %%
print(encoded_docs)


# %%
model=tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=len(encoded_docs),
        output_dim=64,
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])


# %%
history=model.fit(train_X,train_Y,epochs=10,batch_size=32)


# %%
test_loss, test_acc = model.evaluate(test_X,test_Y)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))


