import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split

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


def split_train_test_valid(file_path, train_path, test_path, valid_path):
    df = pd.read_csv(file_path)

    mixed_df = df.sample(frac=1, random_state=42)

    p70_idx = int(.7 * len(mixed_df))
    p85_idx = int(.85 * len(mixed_df))

    train_df, test_df, valid_df = np.split(mixed_df, [p70_idx, p85_idx])

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    valid_df.to_csv(valid_path, index=False)

# def main():
    # creating combined dataset file
    # true_path = 'datasets/kaggle_clement/True.csv'
    # fake_path = 'datasets/kaggle_clement/Fake.csv'
    # save_path = 'datasets/kaggle_clement/dataset.csv'
    #
    # combine_true_fake_datasets(true_path, fake_path, save_path)

    # dividing dataset
    # file_path = 'datasets/kaggle_clement/dataset.csv'
    # train_path = 'datasets/kaggle_clement/train.csv'
    # test_path = 'datasets/kaggle_clement/test.csv'
    # valid_path = 'datasets/kaggle_clement/valid.csv'
    #
    # split_train_test_valid(file_path, train_path, test_path, valid_path)

# if __name__ == "__main__":
#     main()