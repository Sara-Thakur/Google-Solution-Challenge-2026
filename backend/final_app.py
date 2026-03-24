from google import genai
import joblib
import pandas as pd

API_KEY = "AIzaSyA17Z64hSUrgZMC_WeLb2bMlQ-6zFdl0K0"
client = genai.Client(api_key=API_KEY)

try:
    my_bank_model = joblib.load('bank_model.pkl')
    encoders = joblib.load('encoders.pkl')
    print("Files loaded successfully!")
except Exception as e:
    print(f"Error loading files: {e}")
    exit()

def check_loan_and_explain(user_data):
    df_input = pd.DataFrame([user_data])
    
    # Encoding input data
    for col, le in encoders.items():
        if col in df_input.columns:
            df_input[col] = le.transform(df_input[col].astype(str))
            
    prediction = my_bank_model.predict(df_input)
    status = "Approved" if prediction[0] == 0 else "Rejected"
    
    prompt = f"Loan Status: {status}. User Data: {user_data}. Write a 3-line fairness report in English about potential bias."
    
    # Trying multiple model names to fix the 404 error
    models_to_try = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
    response_text = "Gemini could not generate report."

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            response_text = response.text
            break # Success! Exit loop
        except:
            continue # Try next model if this one fails

    print(f"\nSTATUS: {status}")
    print(f"REPORT: {response_text}")

sample_user = {
    'Age': 22, 'Sex': 'female', 'Job': 2, 'Housing': 'rent', 
    'Saving acc': 'little', 'Checking a': 'moderate', 'Credit amount': 5000, 'Duration': 24, 'Purpose': 'education'
}

check_loan_and_explain(sample_user)