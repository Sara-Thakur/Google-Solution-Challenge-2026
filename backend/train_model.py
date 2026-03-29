import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def train_dynamic_model(file_path):
    df = pd.read_csv(file_path)
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    
    target = df.columns[-1] 
    features = [col for col in df.columns if col != target]

    encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df[features]
    y = df[target]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save features list
    joblib.dump(features, 'feature_names.pkl') 
    print(f"✅ AI synced with CSV. Detected columns: {len(features)}")

if __name__ == "__main__":
    train_dynamic_model(r'C:\Users\panka\OneDrive\Desktop\pratibha\bank project\backend\loan_dataset_20000.csv')