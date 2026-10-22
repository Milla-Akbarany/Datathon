import pandas as pd
df = pd.read_csv("Dataset/dataset_A/train.csv").drop(columns=["sample_id", "target"])
print(df.duplicated().sum())