import pandas as pd
from sklearn.model_selection import train_test_split
import os

INPUT_PATH = "data/mushrooms.csv"
OUTPUT_DIR = "data"

def preprocess_and_split(test_size=0.2, random_state=42):
    df = pd.read_csv(INPUT_PATH)

    df['class'] = df['class'].map({'e': 0, 'p': 1})

    X = pd.get_dummies(df.drop('class', axis=1))
    y = df['class']

    X['class'] = y

    train, test = train_test_split(X, test_size=test_size, random_state=random_state, stratify=y)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    test.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    print(f"Saved train ({len(train)}) and test ({len(test)}) to {OUTPUT_DIR}/")

if __name__ == "__main__":
    preprocess_and_split()
