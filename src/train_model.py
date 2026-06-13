import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from data_prep import load_data, preprocess_dataframe

# Load data
df = load_data("data/spam.csv")
df = preprocess_dataframe(df)

X = df['text']
y = df['label']

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
nb_model = MultinomialNB()


# Train
nb_model.fit(X_train, y_train)

# Save models
os.makedirs("model", exist_ok=True)

pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))
pickle.dump(nb_model, open("model/naive_bayes.pkl", "wb"))

print("Models trained and saved successfully!")
