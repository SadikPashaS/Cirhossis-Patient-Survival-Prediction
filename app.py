from flask import Flask, render_template, request
import pandas as pd
import pickle
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load model and scaler
rf_model = pickle.load(open("rf.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Columns to use
num_cols = ['Cholesterol', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage']
cat_cols = ['Drug', 'Ascites', 'Hepatomegaly', 'Spiders']

@app.route('/')
def index():
    return render_template("front.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        return "<h3>Error: No file uploaded.</h3>"

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)

        if 'Status' in df.columns:
            df.drop(columns=['Status'], inplace=True)

        # Handle missing values
        for col in num_cols:
            if col in df.columns:
                df[col].fillna(df[col].mean(), inplace=True)
        for col in cat_cols:
            if col in df.columns:
                df[col].fillna(df[col].mode()[0], inplace=True)

        # One-hot encoding
        df_encoded = pd.get_dummies(df, columns=['Drug', 'Ascites', 'Hepatomegaly', 'Spiders', 'Sex', 'Edema'])

        # Drop 'ID' for prediction
        if 'ID' in df_encoded.columns:
            df_encoded.drop(columns=['ID'], inplace=True)

        # Reindex with training columns
        model_columns = pickle.load(open("model_columns.pkl", "rb"))
        df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

        # Scale features
        df_scaled = scaler.transform(df_encoded)

        # Predict using Random Forest
        rf_preds = rf_model.predict(df_scaled)

        # Map status meaning
        # Convert to binary survival status
        def binary_status(s):
            if s in ['C', 'CL']:
                return "1 (Will Survive)"
            elif s == 'D':
                return "0 (Death)"
            else:
                return "Unknown"

        df['Age'] = (df['Age'] // 365).round(2)
        df['Cirhossis_Prediction'] = [binary_status(s) for s in rf_preds]


        # Output only relevant columns
        output_df = df[['ID', 'Age', 'Cirhossis_Prediction']]

        return render_template("output.html", tables=output_df.to_html(classes='data', index=False), titles=output_df.columns.values)

    except Exception as e:
        return f"<h3>Error: {str(e)}</h3><p>Check your CSV formatting.</p>"

if __name__ == '__main__':
    app.run(debug=True)
