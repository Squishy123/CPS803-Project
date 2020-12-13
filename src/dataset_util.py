import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_true_fake_pct():

    file = "datasets/kaggle_clement/og files/dataset.csv"
    save_file = "datasets_true_fake_pct_plot.png"
    df = pd.read_csv(file)
    total = float(df.shape[0])

    fake_df = df[df.label == 0]
    fake_count = fake_df.shape[0]
    cl_fake_pct = round(fake_count / total, 2) * 100

    cl_true_pct = 100 - cl_fake_pct

    file = "datasets/kaggle_comp/og_files/dataset.csv"
    df = pd.read_csv(file)
    total = float(df.shape[0])

    fake_df = df[df.label == 0]
    fake_count = fake_df.shape[0]
    co_fake_pct = round(fake_count / total, 2) * 100

    co_true_pct = 100 - co_fake_pct

    pct_data = {
        "Dataset": ["Comp", "Clement"],
        "True news": [co_true_pct, cl_true_pct],
        "Fake news": [co_fake_pct, cl_fake_pct]
    }
    pct_df = pd.DataFrame(data=pct_data)

    pct_df.plot(
        x="Dataset",
        kind="barh",
        stacked=True,
        figsize=[7,2.3])

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

if __name__ == "__main__":
    main()