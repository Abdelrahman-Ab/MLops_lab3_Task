import pandas as pd
from sklearn.model_selection import train_test_split
import os

data_path = "data/mushrooms.csv"
df = pd.read_csv(data_path)

df = df.dropna()

df = df.apply(lambda col: col.astype('category').cat.codes if col.dtypes == 'object' else col)

train, test = train_test_split(df, test_size=0.2, random_state=42)

os.makedirs("data", exist_ok=True)

train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)

print("✅ Data preprocessed and saved as train.csv and test.csv")
