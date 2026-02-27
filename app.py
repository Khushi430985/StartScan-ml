from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load('random_forest_model.pkl')

# Feature names in correct order (same as training)
FEATURES = [
    'age_first_funding_year',
    'age_last_funding_year',
    'age_first_milestone_year',
    'age_last_milestone_year',
    'relationships',
    'funding_rounds',
    'funding_total_usd',
    'milestones',
    'avg_participants'
]

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input safely by name (not by order)
        input_data = [float(request.form.get(feature)) for feature in FEATURES]

        # Create DataFrame
        data = pd.DataFrame([input_data], columns=FEATURES)

        # Prediction
        prediction = model.predict(data)[0]

        if prediction == 1:
            result = "Startup is likely to be Acquired"
        else:
            result = "Startup is likely to be Closed"

        return render_template('result.html', prediction_text=result)

    except Exception as e:
        return render_template('result.html', prediction_text="Error in input values")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
    