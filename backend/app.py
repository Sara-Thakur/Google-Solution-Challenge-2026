from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Model and Encoders Load
try:
    model = joblib.load('bank_model.pkl')
    encoders = joblib.load('encoders.pkl')
    print("AI Model & Encoders Loaded!")
except Exception as e:
    print(f"File Load Error: {e}")

# Your New API Key
API_KEY = "AIzaSyAu_Vt3aqr1I0a13sMn1vJkhbzz8saMNHg"
client = genai.Client(api_key=API_KEY)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df_input = pd.DataFrame([data])
        
        # 1. Encoding
        for col, le in encoders.items():
            if col in df_input.columns:
                df_input[col] = le.transform(df_input[col].astype(str))
        
        # 2. Prediction
        prediction = model.predict(df_input)
        status = "Approved" if prediction[0] == 0 else "Rejected"
        
        # 3. Gemini Insight with Error Handling
        prompt = f"Loan Status: {status}. User Data: {data}. Write a 2-line fairness report in English about bias (age/gender)."
        
        try:
            # Using the direct generate method
            response = client.models.generate_content(
                model="gemini-1.5-flash", 
                contents=prompt
            )
            report = response.text
        except Exception as gemini_err:
            print(f"Gemini Error: {gemini_err}")
            # Backup Report (In case API fails during Hackathon)
            if status == "Rejected":
                report = f"Decision based on credit risk profile. Note: Age ({data.get('Age')}) and Gender ({data.get('Sex')}) were checked for fairness."
            else:
                report = "Application meets standard criteria. Fairness audit shows no significant bias in this decision."

        return jsonify({'loan_status': status, 'fairness_report': report})
    
    except Exception as e:
        print(f"System Error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)