import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

# Example dataset (replace with actual health data)
X = np.array([[1, 2, 3, 4], [4, 3, 2, 1], [2, 2, 3, 3], [3, 3, 2, 2]])
y = np.array([0, 1, 0, 1])  # Health labels (0 = healthy, 1 = needs attention)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Save the model to a file
with open('logistic_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model trained and saved as logistic_model.pkl")
