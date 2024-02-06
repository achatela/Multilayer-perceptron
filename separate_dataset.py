import sys
import csv
import pandas as pd

# Normalize the dataframe into a range of 0 to 1
def normalize_dataframe(df):
    for column in df.columns:
        if column != "Diagnosis":
            df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    

def seperate_dataset(filename):
    try:
        df = pd.read_csv(filename)
        df = df.drop(columns="ID")
        normalize_dataframe(df)
        df1 = df.iloc[:40,:]
        df2 = df.iloc[40:,:]
        df1.to_csv("validation_dataset.csv", index=False)
        df2.to_csv("./training/training_dataset.csv", index=False)
    except Exception as e:
        sys.exit(e)

def main():
    if len(sys.argv) != 2 or ".csv" not in sys.argv[1]:
        sys.exit("python3 separate_dataset.py ./path_dataset.csv")
    seperate_dataset(sys.argv[1])

if __name__ == "__main__":
    main()