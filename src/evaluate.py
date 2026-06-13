import pickle
from sklearn.metrics import accuracy_score, classification_report

# Load models
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
nb_model = pickle.load(open("model/naive_bayes.pkl", "rb"))

from data_prep import load_data, preprocess_dataframe

df = load_data("data/spam.csv")
df = preprocess_dataframe(df)

X = vectorizer.transform(df['text'])
y = df['label']

# Prediction
nb_pred = nb_model.predict(X)


# Evaluation
print("Naive Bayes Accuracy:", accuracy_score(y, nb_pred))


print("\nNaive Bayes Report:\n", classification_report(y, nb_pred))

