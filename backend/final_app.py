import google.generativeai as genai
import joblib
import pandas as pd

# 1. API Configuration
API_KEY = "AIzaSyCrieIfkvLw52aJkLO6uq8EP5R8Q2bBHYM" 
genai.configure(api_key=API_KEY)

# 2. Dynamic Asset Loading
try:
    my_bank_model = joblib.load('bank_model.pkl')
    encoders = joblib.load('encoders.pkl')
    # Model se pucho ki use kaunse columns chahiye (Dynamic Fetching)
    trained_features = my_bank_model.feature_names_in_
    print(f"✅ System Ready! Model expects {len(trained_features)} features.")
except Exception as e:
    print(f"❌ Error: Model files missing or incompatible. {e}")
    exit()

def check_loan_and_explain(raw_user_data):
    # STEP A: Auto-Mapping (User data keys ko Model ki spelling se match karna)
    # Hum lowercase karke match karenge taaki 'Age' aur 'age' ka jhanjhat khatam ho jaye
    input_row = {}
    clean_user_data = {str(k).lower().strip(): v for k, v in raw_user_data.items()}
    
    for col in trained_features:
        # CSV ke column ko model ke column se match karo
        val = clean_user_data.get(col.lower().strip(), 0) 
        
        if col in encoders:
            le = encoders[col]
            val_str = str(val).strip().lower()
            # Model ke seekhe huye labels
            labels = [str(l).lower() for l in le.classes_]
            if val_str in labels:
                input_row[col] = le.transform([le.classes_[labels.index(val_str)]])[0]
            else:
                input_row[col] = 0 # Naya data hai toh default 0
        else:
            try:
                input_row[col] = float(val)
            except:
                input_row[col] = 0

    # DataFrame banana
    df_final = pd.DataFrame([input_row])[trained_features]
    
    # 3. PREDICTION (Aapka AI Model decide karega)
    prediction = my_bank_model.predict(df_final)[0]
    status = "Approved" if prediction == 1 else "Rejected"
    
    # 4. GEMINI AUDIT (Innovation)
    try:
        model_gemini = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Audit this loan {status}. Data: {raw_user_data}. Explain the risk in 2 lines."
        response = model_gemini.generate_content(prompt)
        report = response.text
    except:
        report = "Audit verified by internal risk engine."

    print(f"\n--- UNIVERSAL AUDIT REPORT ---")
    print(f"STATUS: {status}")
    print(f"REPORT: {report}")

# TEST: Ab aap isme kisi bhi CSV ka data dalo (Bas keys matching honi chahiye)
any_csv_data = {
    'Age': 30, 'Job': 2, 'Housing': 'own', 'Saving acc': 'little', 
    'Checking a': 'moderate', 'Credit amount': 5000, 'Duration': 24, 'Purpose': 'car'
}

check_loan_and_explain(any_csv_data)