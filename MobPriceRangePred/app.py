from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

app = Flask(__name__)

# Load the data
df = pd.read_csv("Mobile_data.csv")
X = df.drop('price_range', axis=1)
y = df['price_range']

# Train the model or load the saved model
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    # Train the model if not already trained and save it
    model = LogisticRegression()
    model.fit(X, y)
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

class_labels = {0: 'Very Low', 1: 'Low', 2: 'High', 3: 'Very High'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    input_data = [float(data[field]) for field in X.columns]
    input_data = np.array(input_data).reshape(1, -1)

    prediction_class = model.predict(input_data)[0]
    predicted_category = class_labels[prediction_class]

    return jsonify({'predicted_category': predicted_category})

if __name__ == '__main__':
    app.run(debug=True)
