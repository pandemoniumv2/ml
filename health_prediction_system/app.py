from flask import Flask, render_template, request
import numpy as np
from model import predict_health_status

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract parameters from form
        param1 = float(request.form['param1'])
        param2 = float(request.form['param2'])
        param3 = float(request.form['param3'])
        param4 = float(request.form['param4'])
        
        # Create feature array
        features = np.array([[param1, param2, param3, param4]])
        
        # Get prediction from model
        result = predict_health_status(features)
        
        # Render the result page
        return render_template('result.html', prediction=result)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
