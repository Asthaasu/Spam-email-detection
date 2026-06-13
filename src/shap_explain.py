import pickle
import shap

# Load
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
rf_model = pickle.load(open("model/random_forest.pkl", "rb"))

def explain_email(email_text):
    processed = vectorizer.transform([email_text])
    
    explainer = shap.Explainer(rf_model)
    shap_values = explainer(processed)
    
    return shap_values
