from flask import Flask, request, jsonify
import pickle
import pandas as pd
#from werkzeug.utils import secure_filename
import numpy as np
#import os


#app = Flask(__name__)

""" @app.route('/')
def hello_world():
    return 'Hello, World!' """



 # Load your trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from Post request
    data = request.get_json()
    # Make sure the data is in a 2D array if it's a single sample
    features = np.array([data['features']])
    # Make prediction
    prediction = model.predict(features)
    # Return results
    return jsonify({'prediction': prediction.tolist()}) 

if __name__ == "__main__":
    app.run(debug=True) 
