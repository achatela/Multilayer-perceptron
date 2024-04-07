import sys
import csv
import pandas as pd

def normalize_dataframe(df):
    for i, column in enumerate(df.columns):
        if i != 0:
            df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
        else:
            df[column] = df[column].apply(lambda x: 1 if x == "M" else 0)

    

def seperate_dataset(filename):
    try:
        df = pd.read_csv(filename, header=None)
        df = df.fillna(0)
        df = df.drop(columns=0, axis=0)

        normalize_dataframe(df)

        df = df.sample(frac=1)

        df1 = df.iloc[:142,:]
        df2 = df.iloc[142:,:]

        df1.to_csv("data_training.csv", index=False)
        df2.to_csv("data_test.csv", index=False)

    except Exception as e:
        sys.exit(e)

def main():
    if len(sys.argv) != 2 or ".csv" not in sys.argv[1]:
        sys.exit("python3 separate_dataset.py ./path_dataset.csv")
    seperate_dataset(sys.argv[1])

if __name__ == "__main__":
    main()