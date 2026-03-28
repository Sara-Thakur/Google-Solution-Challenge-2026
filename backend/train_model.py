import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

np.random.seed(42)
n_rows = 1000

data = {
    'Age': np.random.randint(18, 70, n_rows),
    'Sex': np.random.choice(['male', 'female'], n_rows),
    'Job': np.random.randint(0, 4, n_rows),
    'Housing': np.random.choice(['own', 'rent', 'free'], n_rows),
    'Saving acc': np.random.choice(['little', 'moderate', 'rich', 'none'], n_rows),
    'Checking a': np.random.choice(['little', 'moderate', 'rich', 'none'], n_rows),
    'Credit amount': np.random.randint(500, 15000, n_rows),
    'Duration': np.random.randint(6, 60, n_rows),
    'Purpose': np.random.choice(['car', 'furniture', 'education', 'business'], n_rows)
}
df = pd.DataFrame(data)

# --- LOGIC FOR RISK (Model training ke liye) ---
# Agar Credit amount kam hai aur Saving account 'rich' hai toh 'good' (0) risk
def define_risk(row):
    if row['Credit amount'] < 4000 or row['Saving acc'] == 'rich':
        return 'good'
    else:
        return 'bad'

df['Risk'] = df.apply(define_risk, axis=1)
df.to_csv('data/complete_bank_data.csv', index=False)

encoders = {}
for col in ['Sex', 'Housing', 'Saving acc', 'Checking a', 'Purpose', 'Risk']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

X = df.drop(['Risk'], axis=1)
y = df['Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

joblib.dump(model, 'bank_model.pkl')
joblib.dump(encoders, 'encoders.pkl')

print(f"Model Accuracy: {accuracy_score(y_test, model.predict(X_test)) * 100:.2f}%")