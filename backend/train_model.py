import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def train_system(file_path):
    # 1. Load Data
    df = pd.read_csv(r'C:\Users\panka\OneDrive\Desktop\pratibha\bank project\backend\loan_dataset_20000.csv')
    
    # 2. SMART CLEANING: Column names se space hatana aur lowercase karna
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    print(f"✅ Recognized Columns: {list(df.columns)}")

    # Target aur Features set karna
    target = 'loan_paid_back' # Yeh column aapki file mein hona zaroori hai
    if target not in df.columns:
        print(f"❌ Error: '{target}' column nahi mila!")
        return

    features = [col for col in df.columns if col != target]

    # 3. Encoding Categorical Data
    encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # 4. Training
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    # 5. SAVE EVERYTHING (Model, Encoders, and Feature Names)
    joblib.dump(model, 'bank_model.pkl')
    joblib.dump(encoders, 'encoders.pkl')
    joblib.dump(features, 'feature_names.pkl') 

    print(f"✅ SUCCESS: Model trained with {model.score(X_test, y_test)*100:.2f}% accuracy!")

# File ka naam check kar lena
train_system('loan_dataset_20000.csv')