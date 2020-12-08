import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text
from wordcloud import WordCloud, STOPWORDS

# Reference
# https://www.geeksforgeeks.org/generating-word-cloud-python/#:~:text=Word%20Cloud%20is%20a%20data,indicates%20its%20frequency%20or%20importance.&text=The%20dataset%20used%20for%20generating,on%20videos%20of%20popular%20artists.

def generate_word_cloud(dataset_file, save_file):
    df = pd.read_csv(dataset_file)
    fake_df = df[df.label == 0].text

    words = ''
    stopwords = text.ENGLISH_STOP_WORDS

    for txt in fake_df:
        tokens = txt.split()

        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        words += " ".join(tokens) + " "

    wordcloud = WordCloud(width=800, height=800,
                          background_color="white",
                          stopwords=text.ENGLISH_STOP_WORDS,
                          min_font_size=10).generate(words)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_file)

def combine_true_fake_datasets(true_path, fake_path, save_path):
    """
    Code to combine the true and fake datasets from the kaggle clement source.
    """

    true_df = pd.read_csv(true_path)
    true_df['label'] = 1

    fake_df = pd.read_csv(fake_path)
    fake_df['label'] = 0

    comb_df = true_df.append(fake_df)

    comb_df.to_csv(save_path, index=False)

def combine_train_valid(train_path, valid_path, save_path):
    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)

    comb = train.append(valid)
    comb.to_csv(save_path, index=False)

def split_train_test_valid(file_path, train_path, test_path, valid_path):
    df = pd.read_csv(file_path)

    mixed_df = df.sample(frac=1, random_state=42)

    p70_idx = int(.7 * len(mixed_df))
    p85_idx = int(.85 * len(mixed_df))

    train_df, test_df, valid_df = np.split(mixed_df, [p70_idx, p85_idx])

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    valid_df.to_csv(valid_path, index=False)

def replace_labels(file_path, save_path, true_label, fake_label):
    df = pd.read_csv(file_path)
    df = df.replace({true_label: 1, fake_label: 0})

    df.to_csv(save_path, index=False)

def clean_dataset(file_path):
    df = pd.read_csv(file_path)
    df = df[df['text'].notnull()]
    df = df[df['label'].notnull()]

    df.to_csv(file_path, index=False)

def main():
    # word cloud
    file = "datasets/kaggle_clement/og files/dataset.csv"
    save_file = "fake_news_word_cloud_clement.png"
    generate_word_cloud(file, save_file)

if __name__ == "__main__":
    main()