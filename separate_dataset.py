import sys
import csv
import pandas as pd

# Normalize the dataframe into a range of 0 to 1
def normalize_dataframe(df):
    maximums = [df[column].max() for column in df.columns]
    minimums = [df[column].min() for column in df.columns]
    for column in df.columns:
        if column != "Diagnosis":
            df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return maximums[1:], minimums[1:]
    

def seperate_dataset(filename):
    try:
        df = pd.read_csv(filename)
        df = df.fillna(0)
        df = df.drop(columns="ID")
        # drop the columns with the name texture2,primeter2,area2,smoothness2,compactness2,concavity2,concave_points2,symmetry2,fractal_dimension2,radius3,texture3,perimeter3,area3,smoothness3,compactness3,concavity3,concave_points3,symmetry3,fractal_dimension3
        # df = df.drop(columns="radius2")
        # df = df.drop(columns="texture2")
        # df = df.drop(columns="perimeter2")
        # df = df.drop(columns="area2")
        # df = df.drop(columns="smoothness2")
        # df = df.drop(columns="compactness2")
        # df = df.drop(columns="concavity2")
        # df = df.drop(columns="concave_points2")
        # df = df.drop(columns="symmetry2")
        # df = df.drop(columns="fractal_dimension2")
        # df = df.drop(columns="radius3")
        # df = df.drop(columns="texture3")
        # df = df.drop(columns="perimeter3")
        # df = df.drop(columns="area3")
        # df = df.drop(columns="smoothness3")
        # df = df.drop(columns="compactness3")
        # df = df.drop(columns="concavity3")
        # df = df.drop(columns="concave_points3")
        # df = df.drop(columns="symmetry3")
        # df = df.drop(columns="fractal_dimension3")

        maximums, minimums = normalize_dataframe(df)
        df1 = df.iloc[:40,:]
        df2 = df.iloc[40:,:]
        df1.to_csv("validation_dataset.csv", index=False)
        df2.to_csv("./training/training_dataset.csv", index=False)
        with open("maximums.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(maximums)
        with open("minimums.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(minimums)
    except Exception as e:
        sys.exit(e)

def main():
    if len(sys.argv) != 2 or ".csv" not in sys.argv[1]:
        sys.exit("python3 separate_dataset.py ./path_dataset.csv")
    seperate_dataset(sys.argv[1])

if __name__ == "__main__":
    main()