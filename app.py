import json
import pickle

from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load the regression model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    """
    Render the homepage with the prediction form.
    """
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    """
    API endpoint for predicting house prices using JSON input.
    """
    try:
        # Parse JSON input
        data = request.get_json()
        if 'data' not in data:
            return jsonify({"error": "Missing 'data' key in the request"}), 400

        # Preprocess input data
        input_array = np.array(list(data['data'].values())).reshape(1, -1)
        transformed_data = scalar.transform(input_array)

        # Predict using the model
        output = regmodel.predict(transformed_data)
        return jsonify({"prediction": output[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle form submission and predict house prices.
    """
    try:
        # Get form input and preprocess it
        form_data = [float(x) for x in request.form.values()]
        final_input = scalar.transform(np.array(form_data).reshape(1, -1))

        # Predict using the model
        output = regmodel.predict(final_input)[0]

        # Render the result on the homepage
        return render_template(
            "home.html",
            prediction_text=f"The House price prediction is {output:.2f}"
        )
    except Exception as e:
        return render_template(
            "home.html",
            prediction_text=f"An error occurred: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)
