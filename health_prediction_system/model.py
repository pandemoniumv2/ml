import pickle
import numpy as np

# Load pre-trained logistic regression model
with open('logistic_model.pkl', 'rb') as model_file:
    logistic_model = pickle.load(model_file)

def predict_health_status(features):
    """
    Function to predict health status based on input features.
    """
    prediction = logistic_model.predict(features)
    
    if prediction == 0:
        return "You are in good health. No need for immediate attention."
    else:
        return "You should visit a doctor or go to the ward."
