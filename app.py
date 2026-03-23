from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # Ye line browser se connection allow karegi

model = joblib.load('bank_model.pkl')
encoders = joblib.load('encoders.pkl')

API_KEY = "AIzaSyA17Z64hSUrgZMC_WeLb2bMlQ-6zFdl0K0"
client = genai.Client(api_key=API_KEY)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df_input = pd.DataFrame([data])
        
        for col, le in encoders.items():
            if col in df_input.columns:
                df_input[col] = le.transform(df_input[col].astype(str))
        
        prediction = model.predict(df_input)
        status = "Approved" if prediction[0] == 0 else "Rejected"
        
        prompt = f"Loan Status: {status}. User Data: {data}. Write a 2-line fairness report about bias."
        
        report = "Analysis not available."
        for model_name in ["gemini-1.5-flash", "gemini-pro"]:
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                report = response.text
                break
            except:
                continue

        return jsonify({'loan_status': status, 'fairness_report': report})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)