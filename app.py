from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and feature names
model = joblib.load("cancer_model.pkl")
feature_names = joblib.load("feature_names.pkl")
    
@app.route("/")
def home():
    return render_template("index.html", feature_names=feature_names)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Debugging: Print received form data
        print("Received form data:", request.form)

        # Extract input values
        input_data = []
        for feature in feature_names:
            if feature not in request.form:
                return render_template("index.html", feature_names=feature_names, error=f"Missing input for: {feature}")
            try:
                input_data.append(float(request.form[feature]))
            except ValueError:
                return render_template("index.html", feature_names=feature_names, error=f"Invalid input: {feature} must be a number")

        # Convert to NumPy array and reshape for prediction
        input_array = np.array(input_data).reshape(1, -1)

        # Predict death rate
        predicted_death_rate = model.predict(input_array)[0]

        return render_template("index.html", feature_names=feature_names, prediction=f"Predicted Death Rate: {predicted_death_rate:.2f}")

    except Exception as e:
        return render_template("index.html", feature_names=feature_names, error=f"Unexpected Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
