
from flask import Flask, render_template, request
import joblib
import pandas as pd
import math

# Create Flask application
app = Flask(__name__)

# Load the saved model and preprocessing objects
artifacts = joblib.load("model.pkl")

model = artifacts["model"]
scaler = artifacts["scaler"]
country_encoder = artifacts["country_encoder"]
continent_encoder = artifacts["continent_encoder"]

# Dropdown values
countries = country_encoder.classes_.tolist()
continents = continent_encoder.classes_.tolist()


# Home page
@app.route("/")
def home():
    return render_template(
        "home.html",
        countries=countries,
        continents=continents
    )


# Prediction page
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get values from the form
        country = request.form["country"]
        continent = request.form["continent"]

        beer = float(request.form["beer_servings"])
        spirit = float(request.form["spirit_servings"])
        wine = float(request.form["wine_servings"])

        # Validate numerical inputs
        values = [beer, spirit, wine]

        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("Serving values must be finite and non-negative.")

        # Validate categories
        if country not in countries:
            raise ValueError("Selected country is invalid.")

        if continent not in continents:
            raise ValueError("Selected continent is invalid.")

        # Encode categorical values
        country_encoded = country_encoder.transform([country])[0]
        continent_encoded = continent_encoder.transform([continent])[0]

        # Prepare input data in the training feature order
        input_data = pd.DataFrame([{
            "country": country_encoded,
            "beer_servings": beer,
            "spirit_servings": spirit,
            "wine_servings": wine,
            "continent": continent_encoded
        }])

        # Scale numerical columns
        numerical_cols = [
            "beer_servings",
            "spirit_servings",
            "wine_servings"
        ]

        input_data[numerical_cols] = scaler.transform(
            input_data[numerical_cols]
        )

        # Generate prediction
        result = model.predict(input_data)[0]

        prediction = round(max(0.0, float(result)), 3)

        # Display result
        print("Model being used:", type(model).__name__)
        print("Model expected features:", model.n_features_in_)
        print("Input shape:", input_data.shape)
        print("Model object ID:", id(model))
        return render_template(
            "res.html",
            prediction_text=(
                "Predicted total litres of pure alcohol: "
                f"{prediction}"
            ),
            prediction=prediction
        )

    except Exception as e:
        # Temporary debugging: show the real error
        print("ACTUAL PREDICTION ERROR:", repr(e))

        return render_template(
            "res.html",
            prediction_text=f"Prediction error: {e}",
            prediction=None
        )


# Run application
if __name__ == "__main__":
    app.run(debug=True)
