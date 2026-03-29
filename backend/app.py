from flask import Flask, request, jsonify
from flask_cors import CORS
import requests, json, re, uuid

app = Flask(__name__)
CORS(app)

API_KEY = "AIzaSyAHz-YjB-gYQqH-RUuwaX8gH5WLggj_prU"

@app.route('/get_features', methods=['GET'])
def get_features():
    features = ["age", "annual_income", "monthly_income", "credit_score", "loan_amount", "interest_rate", "debt_to_income_ratio", "employment_status", "loan_purpose", "delinquency_history"]
    return jsonify({'features': features})

def get_expert_decision(data):
    """AI fail hone par ye function dimaag lagayega"""
    try:
        income = float(data.get('annual_income', 0))
        loan = float(data.get('loan_amount', 0))
        score = float(data.get('credit_score', 0))
        
        if income > (loan * 2) and score > 700:
            return "Approved", "Financials are strong and credit score is healthy."
        elif score < 500:
            return "Rejected", "Credit score is too low for safe lending."
        elif loan > (income * 0.8):
            return "Rejected", "Loan amount is too high compared to annual income."
        else:
            return "Rejected", "Insufficient financial stability detected."
    except:
        return "Rejected", "Invalid data format provided."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_data = request.get_json()
        
        # --- PHASE 1: Try Gemini AI ---
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
        payload = {"contents": [{"parts": [{"text": f"Analyze: {json.dumps(user_data)}. Return JSON: {{'decision': 'Approved/Rejected', 'report': 'Reason'}}"}]}]}
        
        try:
            res = requests.post(url, json=payload, timeout=5) # Sirf 5 sec wait karo
            if res.status_code == 200:
                ai_raw = res.json()['candidates'][0]['content']['parts'][0]['text']
                match = re.search(r'\{.*\}', ai_raw, re.DOTALL)
                if match:
                    ai_data = json.loads(match.group())
                    return jsonify({'loan_status': ai_data['decision'], 'fairness_report': ai_data['report']})
        except:
            pass # Agar AI fail ho jaye toh niche wale Expert System par jao

        # --- PHASE 2: EXPERT SYSTEM (The Savior) ---
        # Agar AI 404 de ya slow ho, toh ye turant result dega
        status, reason = get_expert_decision(user_data)
        return jsonify({
            'loan_status': status, 
            'fairness_report': f"Audit: {reason}"
        })

    except Exception as e:
        return jsonify({'loan_status': 'Error', 'fairness_report': str(e)}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)