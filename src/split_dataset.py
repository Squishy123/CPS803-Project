import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv('../datasets/dataset1.csv')

    # dropping the id column
    df = df.drop(['id'], axis=1)

    train_data, test_data = train_test_split(df, train_size=0.8, random_state=42)

    train_data.to_csv('../datasets/train1.csv', index=False)
    test_data.to_csv('../datasets/test1.csv', index=False)

if __name__ == "__main__":
    main()