import pandas as pd
import re
import string

def load_data(path):
    df = pd.read_csv(path)

    # Keep only required columns
    df = df[['label', 'message']]
    df.columns = ['label', 'text']

    # Convert labels
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    return df


def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text


def preprocess_dataframe(df):
    df['text'] = df['text'].apply(preprocess)
    return df
