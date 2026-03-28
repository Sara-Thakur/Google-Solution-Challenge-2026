from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from google import genai

app = Flask(__name__)
CORS(app)

# --- LOAD ASSETS ---
try:
    model = joblib.load('bank_model.pkl')
    encoders = joblib.load('encoders.pkl')
    trained_features = joblib.load('feature_names.pkl')
    print("✅ All AI Assets Loaded Successfully!")
except Exception as e:
    print(f"❌ Load Error: {e}")

# GEMINI SETUP (Apni Key Yahan Dalein)
API_KEY = "AIzaSy..." 
client = genai.Client(api_key=API_KEY)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        raw_data = request.get_json()
        
        # 1. Terminal mein data print karna taaki hum error pakad sakein
        print(f"\n📥 RECEIVED FROM FRONTEND: {raw_data}")
        
        # Frontend keys ko clean karna
        clean_input = {str(k).strip().lower().replace(' ', '_'): v for k, v in raw_data.items()}
        
        input_row = {}
        for col in trained_features:
            # Agar column frontend se nahi aaya toh hum use 0 maan rahe hain
            val = clean_input.get(col, 0)
            
            if col in encoders:
                clean_val = str(val).strip()
                # Case-insensitive matching for categories
                labels_map = {str(l).lower(): l for l in encoders[col].classes_}
                
                if clean_val.lower() in labels_map:
                    original_label = labels_map[clean_val.lower()]
                    input_row[col] = encoders[col].transform([original_label])[0]
                else:
                    # Agar label bilkul naya hai toh training ka pehla label le lo crash ke bajaye
                    input_row[col] = 0
            else:
                try:
                    input_row[col] = float(val) if val else 0.0
                except:
                    input_row[col] = 0.0

        # 2. Prediction logic
        df_final = pd.DataFrame([input_row])
        # AI decision
        prediction = model.predict(df_final)[0]
        prob = model.predict_proba(df_final)[0]
        
        status = "Approved" if int(prediction) == 1 else "Rejected"
        confidence = round(max(prob) * 100, 2)

        # 3. Gemini Audit
        try:
            prompt = f"Explain why a loan was {status} for this profile: {raw_data}. Be technical and brief."
            response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
            report = response.text
        except:
            report = f"System verdict: {status}. Manual audit required for detailed insights."

        return jsonify({'loan_status': status, 'confidence': confidence, 'fairness_report': report})

    except Exception as e:
        # YAHAN ERROR PRINT HOGA TERMINAL MEIN
        print(f"❌ CRITICAL ERROR: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)